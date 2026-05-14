"""deepMaze CLI entrypoint.

Wires EventBus → MetricsCollector / TrajectoryCollector / TqdmTail → MazeManager.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("agents", "environment", "training", "utils", "web", "config"):
    p = os.path.join(_HERE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from maze import MazeEnvironment, RenderMaze   # noqa: E402
from manager import MazeManager                # noqa: E402
from train import create_agent, train_agent, evaluate_agent, simulate_episode  # noqa: E402
from recorders import (MetricsCollector, TqdmTail, TrajectoryCollector,    # noqa: E402
                       ReplayRecorder)
from viz_events import EventBus                # noqa: E402
from seeding import seed_everything             # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="deepMaze — RL maze playground")
    p.add_argument("--agent_type", choices=["q", "dqn", "ppo"], default="q")
    p.add_argument("--maze_width", type=int, default=10)
    p.add_argument("--maze_height", type=int, default=10)
    p.add_argument("--n_agents", type=int, default=1)
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--generator", choices=["random", "dfs", "open"], default="random")
    p.add_argument("--no_ensure_solvable", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--discount_factor", type=float, default=0.99)
    p.add_argument("--image_path", type=str, default=None,
                   help="Folder containing sprite sheet PNGs (e.g. sprites.png). "
                        "If omitted, placeholder colored tiles are used.")
    p.add_argument("--sprite_files", nargs="*", default=["sprites.png"])
    p.add_argument("--sprite_size", type=int, default=32)
    p.add_argument("--replay_fmt", choices=["webp", "gif", "mp4"], default="webp")
    p.add_argument("--frame_skip", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--policy_snapshot_every", type=int, default=50)
    p.add_argument("--live", action="store_true", help="Single-line live status")
    p.add_argument("--live_web", action="store_true", help="Start web viewer thread")
    p.add_argument("--web_port", type=int, default=8000)
    p.add_argument("--run_id", type=str, default=None)
    return p


def _load_sprites(args) -> list:
    if not args.image_path:
        return RenderMaze.placeholder_sprites(args.sprite_size)
    return RenderMaze.crop_images(
        args.image_path, args.sprite_files,
        sprite_size=args.sprite_size,
        return_indexes=[0, 1, 2, 3, 4],
    )


def run(args):
    seed_everything(args.seed)
    mgr = MazeManager(run_id=args.run_id)
    mgr.save_config(vars(args))

    env = MazeEnvironment(width=args.maze_width, height=args.maze_height,
                          n_agents=args.n_agents, density=args.density,
                          seed=args.seed, generator=args.generator,
                          ensure_solvable=not args.no_ensure_solvable)
    mgr.log(f"Maze: gen={args.generator} solvable={env.is_solvable()}")
    agent_kw = {"discount_factor": args.discount_factor}
    if args.learning_rate is not None:
        agent_kw["learning_rate"] = args.learning_rate
    agent = create_agent(args.agent_type, env, **agent_kw)
    mgr.log(f"Agent: {type(agent).__name__}; "
            f"maze: {args.maze_height}x{args.maze_width}; "
            f"episodes: {args.num_episodes}")

    bus = EventBus()
    metrics = MetricsCollector()
    traj = TrajectoryCollector()
    bus.subscribe(metrics)
    bus.subscribe(traj)
    if args.live or sys.stderr.isatty():
        bus.subscribe(TqdmTail(every=max(1, args.num_episodes // 200)))

    web_thread = None
    if args.live_web:
        from server import start_in_thread  # type: ignore
        web_thread = start_in_thread(bus, mgr, port=args.web_port)
        mgr.log(f"Web viewer started on port {args.web_port}")

    mgr.log("Training...")
    train_agent(env, agent,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                bus=bus,
                policy_snapshot_every=args.policy_snapshot_every,
                emit_steps=True)

    mgr.log("Evaluating...")
    avg_r, avg_l, success = evaluate_agent(env, agent,
                                           num_episodes=args.eval_episodes,
                                           max_steps=args.max_steps)
    mgr.save_results({"avg_reward": avg_r, "avg_length": avg_l,
                      "success_rate": success,
                      "episodes_trained": args.num_episodes})
    mgr.log(f"Eval: avg_reward={avg_r:.3f}  avg_length={avg_l:.1f}  "
            f"success={success*100:.1f}%")

    mgr.save_model(agent)
    mgr.save_curves(metrics.episodes)
    mgr.save_visitation(traj.trajectories, env)

    q_source = agent.q_values if not hasattr(agent, "Q") else dict(agent.Q)
    try:
        mgr.save_policy_heatmap(q_source, env)
    except Exception as e:
        mgr.log(f"Policy heatmap skipped: {e}", "warning")

    # Replay WebP from one greedy episode starting at the canonical Start cell.
    agent.set_deterministic(True)
    try:
        states, positions, _, total = simulate_episode(env, agent, args.max_steps,
                                                      at_start=True)
    finally:
        agent.set_deterministic(False)

    sprites = _load_sprites(args)
    rm = RenderMaze(sprites)
    q_seq = [agent.q_values(s) for s in states] if hasattr(agent, "q_values") else None
    ReplayRecorder(rm).feed(states, positions, q_seq)
    mgr.save_replay(rm, fmt=args.replay_fmt,
                    sprite_size=args.sprite_size,
                    frame_skip=args.frame_skip,
                    max_frames=args.max_frames)
    mgr.log(f"Replay episode reward: {total:.3f}")

    mgr.log(f"Run dir: {mgr.run_dir}")
    if web_thread is not None:
        mgr.log("Web viewer still running; Ctrl-C to exit.")
        try:
            web_thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    run(build_argparser().parse_args())
