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
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for sub in ("agents", "environment", "training", "utils", "config"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from viz_events import EventBus, EpisodeEvent, PolicyEvent, RunEvent, StepEvent  # noqa: E402


STATIC = os.path.join(_HERE, "static")


def _event_to_json(ev) -> str:
    if isinstance(ev, (StepEvent, EpisodeEvent, PolicyEvent, RunEvent)):
        return json.dumps(ev.to_json())
    return json.dumps({"type": "unknown"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(bus: Optional[EventBus] = None, manager=None) -> FastAPI:
    app = FastAPI(title="deepMaze viewer")
    app.state.bus = bus or EventBus()
    app.state.manager = manager
    app.state.runs: Dict[str, dict] = {}

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

            if user_maze is not None:
                m = np.asarray(user_maze, dtype=np.uint8)
                env = maze_mod.MazeEnvironment(
                    width=m.shape[1], height=m.shape[0],
                    generator="open", ensure_solvable=False,
                    seed=body.get("seed"),
                )
                env.maze = m.copy()
                env.start_pos = tuple(map(int, np.argwhere(m == 2)[0]))
                env.treasure_pos = tuple(map(int, np.argwhere(m == 3)[0]))
                env.reset(at_start=True)
            else:
                env = maze_mod.MazeEnvironment(
                    width=body.get("width", 8),
                    height=body.get("height", 8),
                    density=body.get("density", 0.2),
                    generator=body.get("generator", "random"),
                    seed=body.get("seed"),
                )
            agent = train_mod.create_agent(body.get("agent_type", "q"), env,
                                           discount_factor=body.get("gamma", 0.95))
            train_mod.train_agent(env, agent,
                                  num_episodes=body.get("num_episodes", 200),
                                  max_steps=body.get("max_steps", 200),
                                  bus=app.state.bus,
                                  policy_snapshot_every=body.get("policy_snapshot_every", 25))

        t = threading.Thread(target=_train, daemon=True)
        t.start()
        app.state.runs[run_id] = {"thread": t}
        return {"run_id": run_id}

    @app.get("/api/events")
    async def events():
        q = app.state.bus.subscribe_queue(maxsize=4096)

        async def gen():
            loop = asyncio.get_event_loop()
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

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

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
