"""One training session = env → agent → train → eval → bundle.

The single implementation behind both heavy-training surfaces
(scripts/train_runpod.py and notebooks/train_agent.ipynb cell 10), which
used to be ~150-line near-clones — 8 of the 2026-06-10 audit's drift items
came from that pair. Surface-specific behavior (Colab Drive mirroring,
IPython display) plugs in via the `on_showcase` callback and post-return
handling of the bundle directory.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from bundles import module_of, save_agent_model, warm_start
from maze import MazeEnvironment, RenderMaze
from recorders import ReplayRecorder
from seeding import seed_everything
from train import create_agent, evaluate_agent, simulate_episode, train_agent
from viz_events import EpisodeEvent, EventBus


def _fmt_eta(s: float) -> str:
    if s < 90:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def _hr(c: str = "─", w: int = 78) -> str:
    return c * w


def train_session(*,
                  agent_type: str,
                  run_name: str,
                  env_kw: dict,
                  num_episodes: int,
                  max_steps: int,
                  assets_dir: Path,
                  showcase_dir: Path,
                  agent_overrides: dict | None = None,
                  warm_start_path: str | None = None,
                  random_start: bool = True,
                  regenerate_every: int | None = None,
                  eval_episodes: int = 50,
                  eval_regenerate: bool = True,
                  eval_every: int = 0,
                  periodic_eval_episodes: int = 10,
                  seed: int = 0,
                  window: int = 100,
                  print_every: int = 50,
                  showcase_every: int = 500,
                  showcase_sprite: int = 12,
                  showcase_frames: int = 300,
                  config_extra: dict | None = None,
                  on_showcase: Callable[[Path], None] | None = None,
                  log_mlflow: bool = True,
                  mode_label: str = "session") -> dict:
    """Run one full train→eval→bundle cycle; returns the result dict both
    surfaces expose (paths, eval metrics, best-snapshot info).

    When `log_mlflow` is true a new MLflow run is opened (caller sets the
    tracking URI / experiment); params, per-episode and periodic metrics,
    and the bundle artifacts are logged.
    """
    mlflow = None
    if log_mlflow:
        import mlflow  # local import keeps test runs mlflow-free

    overrides = dict(agent_overrides or {})

    print()
    print(_hr("━"))
    print(f"  {agent_type.upper()}  —  {run_name}  "
          f"({env_kw.get('width')}×{env_kw.get('height')}, "
          f"{env_kw.get('n_treasures', 1)} treasures)")
    print(_hr("━"))

    seed_everything(seed)
    env = MazeEnvironment(**env_kw)
    agent = create_agent(agent_type, env, **overrides)

    if warm_start_path:
        try:
            warm_start(agent, warm_start_path)
            print(f"  warm start  : {warm_start_path}")
        except Exception as e:
            print(f"  warm start FAILED ({e}) — training from scratch")

    print(f"  mode        : {mode_label}")
    print(f"  agent       : {type(agent).__name__}")
    print(f"  budget      : {num_episodes} eps  max_steps={max_steps}")
    print(f"  regen every : {regenerate_every or 'off'}")
    print(f"  overrides   : {overrides or '(none — repo defaults)'}")
    print(_hr())

    showcase_dir = Path(showcase_dir)
    showcase_dir.mkdir(parents=True, exist_ok=True)
    sprites = RenderMaze.placeholder_sprites(showcase_sprite)

    def render_snapshot(ep: int) -> Path:
        agent.set_deterministic(True)
        try:
            _, positions, _, _ = simulate_episode(env, agent,
                                                  max_steps=max_steps,
                                                  at_start=True)
        finally:
            agent.set_deterministic(False)
        full = [env.maze.copy() for _ in positions]
        rm = RenderMaze(sprites)
        ReplayRecorder(rm).feed(full, positions, None)
        out = showcase_dir / f"ep{ep:05d}.webp"
        rm.save(str(out), fmt="webp", sprite_size=showcase_sprite,
                max_frames=showcase_frames)
        if on_showcase is not None:
            on_showcase(out)
        return out

    bus = EventBus()
    from collections import deque
    reward_buf = deque(maxlen=window)
    length_buf = deque(maxlen=window)
    success_buf = deque(maxlen=window)
    t0 = time.time()

    def on_ep(ev: EpisodeEvent):
        if mlflow is not None:
            mlflow.log_metrics(
                {"episode_reward": ev.total_reward,
                 "episode_length": ev.length,
                 "epsilon": ev.epsilon},
                step=ev.episode,
            )
        reward_buf.append(ev.total_reward)
        length_buf.append(ev.length)
        success_buf.append(1 if ev.success else 0)

        elapsed = time.time() - t0
        eps_per_s = (ev.episode + 1) / max(elapsed, 1e-6)
        eta = (num_episodes - ev.episode - 1) / max(eps_per_s, 1e-6)

        if ev.episode % print_every == 0 or ev.episode == num_episodes - 1:
            avg_r = sum(reward_buf) / len(reward_buf)
            succ = 100.0 * sum(success_buf) / len(success_buf)
            extra = f" loss={ev.loss:.4f}" if ev.loss is not None else ""
            print(
                f"  ep {ev.episode:>5}/{num_episodes}  "
                f"R={ev.total_reward:+6.2f}  R̄{window}={avg_r:+6.2f}  "
                f"succ%={succ:5.1f}  "
                f"len={ev.length:>4}  ε={ev.epsilon:.3f}{extra}  "
                f"[{eps_per_s:5.1f} ep/s  ETA {_fmt_eta(eta)}]",
                flush=True,
            )

        if showcase_every and ev.episode > 0 and ev.episode % showcase_every == 0:
            snap = render_snapshot(ev.episode)
            print(f"  ▣ SHOWCASE @ {ev.episode}  →  {snap}", flush=True)

    bus.subscribe(on_ep)

    best = {"succ": -1.0, "sd": None, "episode": None}

    def on_eval(ep: int, m: dict):
        if mlflow is not None:
            mlflow.log_metrics({f"periodic_{k}": v for k, v in m.items()},
                               step=ep)
        print(f"  ◈ eval @ {ep}: succ={m['success_rate']:.1%} "
              f"R̄={m['mean_reward']:+.2f} "
              f"revisit={m.get('revisit_rate', 0.0):.0%}", flush=True)
        if m["success_rate"] > best["succ"]:
            best["succ"] = m["success_rate"]
            best["episode"] = ep
            module = module_of(agent)
            if module is not None:
                best["sd"] = {k: v.detach().cpu().clone()
                              for k, v in module.state_dict().items()}

    eval_every = eval_every or max(50, num_episodes // 10)

    params = dict(
        agent_type=agent_type, run_name=run_name,
        num_episodes=num_episodes, max_steps=max_steps,
        random_start=random_start, regenerate_every=regenerate_every,
        eval_regenerate=eval_regenerate,
        warm_start_from=warm_start_path or "",
        **{k: v for k, v in env_kw.items() if k != "seed"},
        seed=seed,
        **overrides,
    )

    def _run_training():
        train_agent(env, agent, num_episodes=num_episodes,
                    max_steps=max_steps, bus=bus,
                    random_start=random_start,
                    regenerate_every=regenerate_every,
                    eval_every=eval_every,
                    eval_episodes=periodic_eval_episodes,
                    on_eval=on_eval)
        extra: dict = {}
        return (*evaluate_agent(env, agent, num_episodes=eval_episodes,
                                max_steps=max_steps,
                                regenerate_each=eval_regenerate,
                                metrics_out=extra), extra)

    run_id = None
    if mlflow is not None:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mean_r, mean_l, succ, final_extra = _run_training()
            mlflow.log_metrics({
                "eval_mean_reward": mean_r,
                "eval_mean_length": mean_l,
                "eval_success_rate": succ,
                "eval_revisit_rate": final_extra.get("revisit_rate", 0.0),
            })
            run_id = run.info.run_id
    else:
        mean_r, mean_l, succ, final_extra = _run_training()

    print()
    print(_hr("━"))
    print(f"  ✓ done in {_fmt_eta(time.time() - t0)}  "
          f"eval: R={mean_r:+.2f}  succ={succ:.1%}  "
          f"revisit={final_extra.get('revisit_rate', 0.0):.0%}")
    print(_hr("━"))

    # ---- bundle export -------------------------------------------------
    out_dir = Path(assets_dir) / run_name
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)
    config = dict(
        agent_type=agent_type, net=overrides.get("net"),
        maze_width=env_kw.get("width"), maze_height=env_kw.get("height"),
        n_agents=1, density=env_kw.get("density", 0.2),
        generator=env_kw.get("generator", "random"),
        no_ensure_solvable=False,
        n_lava=env_kw.get("n_lava", 0),
        lava_reward=env_kw.get("lava_reward", -1.0),
        bump_penalty=env_kw.get("bump_penalty", -0.1),
        aux_features=env_kw.get("aux_features", False),
        reward_shaping=env_kw.get("reward_shaping", False),
        partial=env_kw.get("partial_view"),
        n_treasures=env_kw.get("n_treasures", 1),
        collect_all=env_kw.get("collect_all", False),
        seed=seed, num_episodes=num_episodes, max_steps=max_steps,
        eval_episodes=eval_episodes,
        learning_rate=None, discount_factor=0.99,
        image_path=None, sprite_files=["sprites.png"], sprite_size=32,
        replay_fmt="webp", frame_skip=1, max_frames=None,
        policy_snapshot_every=50, live=False, live_web=False, web_port=8000,
        run_id=run_id, run_name=run_name,
        random_start=random_start, resume=None,
        eval_maze="same", eval_seeds=1,
        agent_hp=overrides,
        **(config_extra or {}),
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    model_path = save_agent_model(agent, out_dir)
    best_path = None
    if best["sd"] is not None:
        import torch
        best_path = out_dir / "model.best.pt"
        torch.save(best["sd"], best_path)
        print(f"  best snapshot: succ={best['succ']:.1%} @ ep {best['episode']}")

    final_snap = render_snapshot(num_episodes)
    shutil.copy(final_snap, out_dir / "viz" / "replay.webp")

    if mlflow is not None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(out_dir), artifact_path=f"assets/{run_name}")

    print(f"  bundle → {out_dir}")
    return {
        "agent_type": agent_type, "run_name": run_name, "run_id": run_id,
        "eval_success_rate": succ, "eval_mean_reward": mean_r,
        "eval_revisit_rate": final_extra.get("revisit_rate", 0.0),
        "best_success_rate": best["succ"],
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "best_model_path": str(best_path) if best_path else None,
    }
