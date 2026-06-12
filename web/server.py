"""FastAPI viewer for deepMaze.

Two modes:
  - Embedded: spawned from main.py via `start_in_thread(bus, mgr)`; SSE
    streams events from the live training bus.
  - Standalone: `python web/server.py` lets you draw a maze, pick an
    agent, kick off a training run on a background thread, watch SSE.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for sub in ("agents", "environment", "training", "utils", "web", "config"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from viz_events import (  # noqa: E402
    EpisodeEvent,
    EvalEvent,
    EventBus,
    PolicyEvent,
    RunEvent,
    StepEvent,
)

STATIC = os.path.join(_HERE, "static")


def _find_model_file(d: str) -> str | None:
    """Prefer best-checkpoint over final; .pt over .pkl."""
    for fn in ("model.best.pt", "model.pt", "model.best.pkl", "model.pkl"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    return None


def _load_model_into(agent, path: str) -> None:
    """Restore agent state from a saved file (bundles.warm_start; also syncs
    the target net, which is harmless for greedy inference)."""
    from bundles import warm_start
    warm_start(agent, path)


def _render_detail(name: str, results: dict, artifacts: list[str]) -> str:
    """Server-rendered run detail page when no static viz/report.html exists."""
    import html
    from urllib.parse import quote
    safe_name = html.escape(name)
    safe_url_name = quote(name, safe="")
    parts = ["<html><head><meta charset='utf-8'>",
             "<link rel='stylesheet' href='/static/styles.css'>",
             f"<title>{safe_name}</title></head><body class='app'>",
             "<header class='topbar'><h1>deepMaze</h1>",
             "<nav><a href='/'>Train</a><a href='/runs' class='active'>Runs</a>",
             "<a href='/memory'>Memory</a></nav></header>",
             "<main class='detail'>",
             f"<h2 class='detail-full'>{safe_name}</h2>"]
    if results:
        summary = "\n".join(
            f"{html.escape(str(k))}: {html.escape(str(v))}"
            for k, v in results.items() if not isinstance(v, (list, dict))
        )
        parts.append(f"<pre class='summary detail-full'>{summary}</pre>")
    for a in artifacts:
        safe_artifact = html.escape(a)
        url = f"/api/runs/{safe_url_name}/file/{quote(a, safe='')}"
        if a.endswith((".webp", ".gif", ".mp4", ".png")):
            parts.append(f"<div><h2>{safe_artifact}</h2><img src='{url}' alt='{safe_artifact}'></div>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def _event_to_json(ev) -> str:
    if isinstance(ev, StepEvent):
        payload = ev.to_json_full() if ev.step == 0 else ev.to_json_delta()
        return json.dumps(payload)
    if isinstance(ev, (EpisodeEvent, EvalEvent, PolicyEvent, RunEvent)):
        return json.dumps(ev.to_json())
    return json.dumps({"type": "unknown"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(bus: EventBus | None = None, manager=None) -> FastAPI:
    app = FastAPI(title="deepMaze viewer")
    app.state.bus = bus or EventBus()
    app.state.manager = manager
    app.state.runs: dict[str, dict] = {}

    cors = os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors if cors != ["*"] else ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        from web.otel import instrument as _otel_instrument
        _otel_instrument(app)
    except ImportError:
        pass

    if os.path.isdir(STATIC):
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/health")
    def health():
        return {"ok": True, "embedded": manager is not None}

    @app.post("/api/runs")
    async def start_run(req: Request):
        """Start a training run.

        Body keys:
          maze: 2D int array — if present, used verbatim (must contain START=2
                and EXIT=3); generator is bypassed.
          width, height, density, generator: only when no maze supplied.
          agent_type: 'q' | 'dqn' | 'ppo'
          num_episodes, max_steps, gamma, seed
        """
        body = await req.json()
        user_maze = body.get("maze")
        if user_maze is not None:
            m = np.asarray(user_maze, dtype=np.uint8)
            if m.ndim != 2 or m.shape[0] < 5 or m.shape[1] < 5:
                raise HTTPException(400, "maze must be 2D with min 5x5")
            if not (m == 2).any():
                raise HTTPException(400, "maze missing START cell (value 2)")
            if not (m == 3).any():
                raise HTTPException(400, "maze missing EXIT cell (value 3)")
        run_id = str(uuid.uuid4())[:8]

        def _train():
            import importlib
            maze_mod = importlib.import_module("maze")
            train_mod = importlib.import_module("train")
            seeding_mod = importlib.import_module("seeding")
            seeding_mod.seed_everything(body.get("seed"))

            feature_kw = dict(
                reward_shaping=bool(body.get("reward_shaping", False)),
                bump_penalty=float(body.get("bump_penalty", -0.1)),
                collect_all=bool(body.get("collect_all", False)),
            )
            if user_maze is not None:
                m = np.asarray(user_maze, dtype=np.uint8)
                env = maze_mod.MazeEnvironment(
                    width=m.shape[1], height=m.shape[0],
                    generator="open", ensure_solvable=False,
                    partial_view=body.get("partial"),
                    seed=body.get("seed"), **feature_kw,
                )
                env.maze = m.copy()
                env.start_pos = tuple(map(int, np.argwhere(m == 2)[0]))
                env.treasure_positions = [tuple(map(int, p))
                                          for p in np.argwhere(m == 3)]
                env.reset(at_start=True)
            else:
                env = maze_mod.MazeEnvironment(
                    width=body.get("width", 8),
                    height=body.get("height", 8),
                    density=body.get("density", 0.2),
                    generator=body.get("generator", "random"),
                    n_treasures=body.get("n_treasures", 1),
                    n_lava=body.get("n_lava", 0),
                    partial_view=body.get("partial"),
                    seed=body.get("seed"), **feature_kw,
                )
            agent_kw = {"discount_factor": body.get("gamma", 0.95)}
            if body.get("agent_type") in ("drqn", "dtqn"):
                agent_kw["learn_every"] = int(body.get("learn_every", 1))
            agent = train_mod.create_agent(body.get("agent_type", "q"), env,
                                           **agent_kw)
            max_steps = body.get("max_steps", 200)
            train_mod.train_agent(
                env, agent,
                num_episodes=body.get("num_episodes", 200),
                max_steps=max_steps,
                bus=app.state.bus,
                policy_snapshot_every=body.get("policy_snapshot_every", 25),
                random_start=bool(body.get("random_start", False)),
                should_stop=lambda: app.state.runs.get(run_id, {}).get("stop"),
            )
            # Victory lap: stream the trained agent playing greedily so the
            # user immediately SEES the result of training.
            if not app.state.runs.get(run_id, {}).get("stop"):
                app.state.bus.publish(RunEvent(kind="replay",
                                               info={"run_id": run_id}))
                agent.set_deterministic(True)
                for ep in range(int(body.get("replay_episodes", 2))):
                    train_mod.simulate_episode_streaming(
                        env, agent, app.state.bus,
                        episode=ep, max_steps=max_steps, at_start=True)
            app.state.bus.publish(RunEvent(kind="end", info={"run_id": run_id}))

        t = threading.Thread(target=_train, daemon=True)
        t.start()
        app.state.runs[run_id] = {"thread": t, "stop": False}
        return {"run_id": run_id}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        entry = app.state.runs.get(run_id)
        if entry is None:
            raise HTTPException(404, run_id)
        entry["stop"] = True
        return {"cancelled": True, "run_id": run_id}

    @app.delete("/api/runs/{name}")
    def delete_run(name: str):
        import shutil
        from pathlib import Path
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "invalid run name")
        base = Path(os.getcwd(), "maze_rl_runs").resolve()
        run_dir = (base / name).resolve()
        if run_dir.parent != base or not run_dir.is_dir():
            raise HTTPException(404, name)
        shutil.rmtree(run_dir)
        return {"deleted": name}

    @app.get("/api/events")
    async def events():
        q = app.state.bus.subscribe_queue(maxsize=4096)

        async def gen():
            loop = asyncio.get_event_loop()
            try:
                while True:
                    try:
                        ev = await loop.run_in_executor(None, q.get, True, 30)
                    except Exception:
                        yield ": keepalive\n\n"
                        continue
                    if ev is None:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {_event_to_json(ev)}\n\n"
                    if isinstance(ev, RunEvent) and ev.kind == "end":
                        yield "event: end\ndata: {}\n\n"
                        break
            finally:
                app.state.bus.unsubscribe_queue(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/maze/generate")
    async def maze_generate(req: Request):
        """Return a server-generated maze. Used by the Web editor."""
        import importlib
        body = await req.json()
        width = int(body.get("width", 10))
        height = int(body.get("height", 10))
        density = float(body.get("density", 0.2))
        n_lava = int(body.get("n_lava", 0))
        n_treasures = int(body.get("n_treasures", 1))
        generator = body.get("generator", "dfs")
        if not (5 <= width <= 50 and 5 <= height <= 50):
            raise HTTPException(400, "width/height must be in [5, 50]")
        if not (0.0 <= density <= 0.6):
            raise HTTPException(400, "density must be in [0.0, 0.6]")
        if not (0 <= n_lava <= 100):
            raise HTTPException(400, "n_lava must be in [0, 100]")
        if not (1 <= n_treasures <= 20):
            raise HTTPException(400, "n_treasures must be in [1, 20]")
        if generator not in ("open", "dfs", "random"):
            raise HTTPException(400, "generator must be open|dfs|random")
        maze_mod = importlib.import_module("maze")
        env = maze_mod.MazeEnvironment(
            width=width, height=height, density=density,
            generator=generator, n_lava=n_lava, n_treasures=n_treasures,
            seed=body.get("seed"),
        )
        return {"maze": env.maze.tolist(),
                "start": list(env.start_pos),
                "treasures": [list(p) for p in env.treasure_positions]}

    @app.get("/api/models")
    def list_models():
        """Pretrained models: assets/* and maze_rl_runs/run_* with a model file."""
        cwd = os.getcwd()
        items = []
        for source, base in (("asset", "assets"), ("run", "maze_rl_runs")):
            base_path = os.path.join(cwd, base)
            if not os.path.isdir(base_path):
                continue
            for name in sorted(os.listdir(base_path), reverse=(source == "run")):
                d = os.path.join(base_path, name)
                if not os.path.isdir(d):
                    continue
                cfg_path = os.path.join(d, "config.json")
                model_path = _find_model_file(d)
                if not (os.path.exists(cfg_path) and model_path):
                    continue
                try:
                    cfg = json.loads(open(cfg_path).read())
                except Exception:
                    continue
                items.append({
                    "source": source,
                    "name": name,
                    "agent_type": cfg.get("agent_type"),
                    "model_file": os.path.basename(model_path),
                })
        return {"models": items}

    @app.post("/api/inference")
    async def inference(req: Request):
        """Load a pretrained model and stream a greedy episode via SSE."""
        body = await req.json()
        source = body.get("source", "run")
        name = body["name"]
        cwd = os.getcwd()
        base = os.path.join(cwd, "assets" if source == "asset" else "maze_rl_runs", name)
        if not os.path.isdir(base):
            raise HTTPException(404, f"{source}/{name} not found")
        cfg_path = os.path.join(base, "config.json")
        model_path = _find_model_file(base)
        if not (os.path.exists(cfg_path) and model_path):
            raise HTTPException(400, f"{name} missing config.json or model file")
        cfg = json.loads(open(cfg_path).read())

        def _run():
            import importlib as _il
            maze_mod = _il.import_module("maze")
            train_mod = _il.import_module("train")
            seeding_mod = _il.import_module("seeding")
            seeding_mod.seed_everything(body.get("seed"))

            maze_source = body.get("maze_source", "trained")
            seed = (cfg.get("seed") if maze_source == "trained"
                    else (body.get("seed") or (cfg.get("seed") or 0) + 1))
            env_kw = dict(
                width=cfg.get("maze_width", 10),
                height=cfg.get("maze_height", 10),
                density=cfg.get("density", 0.2),
                generator=cfg.get("generator", "random"),
                n_lava=cfg.get("n_lava", 0),
                lava_reward=cfg.get("lava_reward", -1.0),
                bump_penalty=cfg.get("bump_penalty", -0.1),
                # aux changes the obs shape — required or load_state_dict
                # mismatches against the checkpoint
                aux_features=cfg.get("aux_features", False),
                partial_view=cfg.get("partial"),
                n_treasures=cfg.get("n_treasures", 1),
                collect_all=cfg.get("collect_all", False),
                seed=seed,
            )
            env = maze_mod.MazeEnvironment(**env_kw)
            if maze_source == "custom" and body.get("custom_maze"):
                import numpy as _np
                m = _np.asarray(body["custom_maze"], dtype=_np.uint8)
                if m.shape == env.maze.shape:
                    env.maze = m.copy()
                    env.reset(at_start=True)

            # Effective hyperparameter overrides from training (architecture
            # sizes etc.); create_agent drops unknown keys, so stale entries
            # from old bundles are safe.
            agent_kw = dict(cfg.get("agent_hp") or {})
            if cfg.get("net"):
                agent_kw["net"] = cfg["net"]
            agent = train_mod.create_agent(cfg["agent_type"], env, **agent_kw)
            _load_model_into(agent, model_path)
            agent.set_deterministic(True)

            episodes = int(body.get("episodes", 1))
            max_steps = int(body.get("max_steps", cfg.get("max_steps", 200)))
            for ep in range(episodes):
                train_mod.simulate_episode_streaming(
                    env, agent, app.state.bus, episode=ep,
                    max_steps=max_steps, at_start=True,
                )
            app.state.bus.publish(RunEvent(kind="end", info={"mode": "inference"}))

        threading.Thread(target=_run, daemon=True).start()
        return {"started": True, "name": name, "source": source}

    @app.get("/runs")
    def runs_page():
        return FileResponse(os.path.join(STATIC, "runs.html"))

    @app.get("/runs/{name}")
    def run_detail_page(name: str):
        # Always serve the (possibly lazy-generated) report.html via the file
        # endpoint. Falls back to a server-rendered detail page only if
        # generation itself fails.
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "invalid run name")
        run_dir = os.path.join(os.getcwd(), "maze_rl_runs", name)
        report = os.path.join(run_dir, "viz", "report.html")
        if not os.path.exists(report) and os.path.isdir(run_dir):
            try:
                from pathlib import Path

                from report import write_html_report
                write_html_report(Path(run_dir))
            except Exception:
                pass
        if os.path.exists(report):
            return FileResponse(report)
        if not os.path.isdir(run_dir):
            raise HTTPException(404, name)
        results = {}
        rp = os.path.join(run_dir, "results.json")
        if os.path.exists(rp):
            try: results = json.loads(open(rp).read())
            except Exception: pass
        artifacts = []
        viz_dir = os.path.join(run_dir, "viz")
        if os.path.isdir(viz_dir):
            artifacts = sorted(os.listdir(viz_dir))
        html = _render_detail(name, results, artifacts)
        from fastapi.responses import HTMLResponse
        return HTMLResponse(html)

    @app.get("/memory")
    def memory_page():
        return FileResponse(os.path.join(STATIC, "memory.html"))

    @app.get("/api/runs/list")
    def list_runs():
        """Enumerate persisted runs from maze_rl_runs/, newest first."""
        base = os.path.join(os.getcwd(), "maze_rl_runs")
        if not os.path.isdir(base):
            return {"runs": []}
        items = []
        for name in sorted(os.listdir(base), reverse=True):
            run_dir = os.path.join(base, name)
            if not os.path.isdir(run_dir):
                continue
            results_path = os.path.join(run_dir, "results.json")
            summary = {}
            if os.path.exists(results_path):
                try:
                    summary = json.loads(open(results_path).read())
                except Exception:
                    pass
            viz = os.path.join(run_dir, "viz")
            artifacts = (sorted(os.listdir(viz)) if os.path.isdir(viz) else [])
            items.append({"name": name, "summary": summary,
                          "artifacts": artifacts})
        return {"runs": items}

    @app.get("/api/runs/{name}/file/{artifact}")
    def run_file(name: str, artifact: str):
        from pathlib import Path
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "invalid run name")
        if "/" in artifact or "\\" in artifact or ".." in artifact:
            raise HTTPException(400, "invalid artifact name")
        base = Path(os.getcwd(), "maze_rl_runs").resolve()
        viz_dir = (base / name / "viz").resolve()
        if base not in viz_dir.parents and viz_dir != base:
            raise HTTPException(400, "invalid path")
        f = (viz_dir / artifact).resolve()
        if viz_dir not in f.parents:
            raise HTTPException(400, "invalid path")
        if artifact == "report.html" and not f.exists() and viz_dir.parent.is_dir():
            try:
                from report import write_html_report
                write_html_report(viz_dir.parent)
            except Exception as e:
                raise HTTPException(500, f"report generation failed: {e}") from e
        if not f.exists():
            raise HTTPException(404, str(f))
        return FileResponse(str(f))

    @app.get("/api/artifacts")
    def artifacts():
        if app.state.manager is None:
            return JSONResponse({"available": False})
        viz = app.state.manager.viz_dir()
        files = sorted(p.name for p in viz.iterdir()) if viz.exists() else []
        return {"available": True, "files": files,
                "run_dir": str(app.state.manager.run_dir)}

    @app.get("/api/artifact/{name}")
    def artifact(name: str):
        if app.state.manager is None:
            raise HTTPException(404, "no manager attached")
        f = app.state.manager.viz_dir() / name
        if not f.exists():
            raise HTTPException(404, str(f))
        return FileResponse(str(f))

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def start_in_thread(bus: EventBus, manager, port: int = 8000) -> threading.Thread:
    app = create_app(bus=bus, manager=manager)

    def _run():
        config = uvicorn.Config(app, host="127.0.0.1", port=port,
                                log_level="warning", access_log=False)
        uvicorn.Server(config).run()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
