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

from manager import MazeManager  # noqa: E402
from maze import MazeEnvironment, RenderMaze  # noqa: E402
from recorders import MetricsCollector, ReplayRecorder, TqdmTail, TrajectoryCollector  # noqa: E402
from seeding import seed_everything  # noqa: E402
from train import (  # noqa: E402
    create_agent,
    default_max_steps,
    evaluate_agent,
    simulate_episode,
    train_agent,
)
from viz_events import EventBus  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="deepMaze — RL maze playground")
    p.add_argument("--agent_type",
                   choices=["q", "dqn", "ppo", "drqn", "dtqn"], default="q")
    p.add_argument("--net", choices=["mlp", "cnn"], default=None,
                   help="Network backbone for DQN/PPO; overrides hyperparam default.")
    p.add_argument("--maze_width", type=int, default=10)
    p.add_argument("--maze_height", type=int, default=10)
    p.add_argument("--n_agents", type=int, default=1)
    p.add_argument("--density", type=float, default=0.2)
    p.add_argument("--generator", choices=["random", "dfs", "open"], default="random")
    p.add_argument("--no_ensure_solvable", action="store_true")
    p.add_argument("--n_lava", type=int, default=0,
                   help="Number of LAVA cells (terminal -1) off the shortest path.")
    p.add_argument("--lava_reward", type=float, default=-1.0)
    p.add_argument("--bump_penalty", type=float, default=-0.1,
                   help="Reward for bumping a wall (use -0.01 on big mazes).")
    p.add_argument("--partial", type=int, default=None,
                   help="Egocentric (2K+1)x(2K+1) window; default full-view.")
    p.add_argument("--n_treasures", type=int, default=1)
    p.add_argument("--collect_all", action="store_true",
                   help="Episode ends only after ALL treasures collected.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Step budget per episode (default: 3*(w+h)*n_treasures "
                        "for collect_all, 3*(w+h) otherwise).")
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
    p.add_argument("--run_name", type=str, default=None,
                   help="Custom run dir suffix (overrides timestamp).")
    p.add_argument("--random_start", action="store_true",
                   help="Sample agent's reset position each episode (default: Start).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a saved model.{pt,pkl} to load before training.")
    p.add_argument("--eval_maze", choices=["same", "fresh"], default="same")
    p.add_argument("--eval_seeds", type=int, default=1,
                   help="Held-out eval averages over this many fresh mazes.")
    return p


def _resume_agent(agent, path: str, mgr) -> None:
    """Load saved state into a freshly-created agent."""
    import pickle

    import torch
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            agent.Q.update(pickle.load(f))
    else:
        module = getattr(agent, "model", None) or getattr(agent, "ac", None)
        module.load_state_dict(torch.load(path, map_location=getattr(agent, "device", "cpu")))
    mgr.log(f"Resumed agent state from {path}")


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
    mgr = MazeManager(run_id=args.run_name or args.run_id)
    mgr.save_config(vars(args))

    env = MazeEnvironment(width=args.maze_width, height=args.maze_height,
                          n_agents=args.n_agents, density=args.density,
                          seed=args.seed, generator=args.generator,
                          ensure_solvable=not args.no_ensure_solvable,
                          n_lava=args.n_lava, lava_reward=args.lava_reward,
                          bump_penalty=args.bump_penalty,
                          partial_view=args.partial,
                          n_treasures=args.n_treasures,
                          collect_all=args.collect_all)
    if args.max_steps is None:
        args.max_steps = default_max_steps(env)
        mgr.save_config(vars(args))  # re-dump with the computed budget
    mgr.log(f"Maze: gen={args.generator} solvable={env.is_solvable()} "
            f"max_steps={args.max_steps}")
    agent_kw = {"discount_factor": args.discount_factor}
    if args.learning_rate is not None:
        agent_kw["learning_rate"] = args.learning_rate
    if args.net is not None:
        agent_kw["net"] = args.net
    agent = create_agent(args.agent_type, env, **agent_kw)
    if args.resume:
        _resume_agent(agent, args.resume, mgr)
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
                emit_steps=True,
                random_start=args.random_start)

    mgr.log("Evaluating...")
    avg_r, avg_l, success = evaluate_agent(env, agent,
                                           num_episodes=args.eval_episodes,
                                           max_steps=args.max_steps)
    results = {"avg_reward": avg_r, "avg_length": avg_l,
               "success_rate": success,
               "episodes_trained": args.num_episodes}
    mgr.log(f"Eval: avg_reward={avg_r:.3f}  avg_length={avg_l:.1f}  "
            f"success={success*100:.1f}%")

    if args.eval_maze == "fresh" and args.eval_seeds > 0:
        held_out = []
        for k in range(args.eval_seeds):
            seed_k = (args.seed or 0) + 1 + k
            env_k = MazeEnvironment(width=args.maze_width, height=args.maze_height,
                                    n_agents=args.n_agents, density=args.density,
                                    seed=seed_k, generator=args.generator,
                                    ensure_solvable=not args.no_ensure_solvable,
                                    n_lava=args.n_lava, lava_reward=args.lava_reward,
                                    bump_penalty=args.bump_penalty,
                                    partial_view=args.partial,
                                    n_treasures=args.n_treasures,
                                    collect_all=args.collect_all)
            r_k, l_k, s_k = evaluate_agent(env_k, agent,
                                           num_episodes=args.eval_episodes,
                                           max_steps=args.max_steps)
            held_out.append({"seed": seed_k, "avg_reward": r_k,
                             "avg_length": l_k, "success_rate": s_k})
        results["held_out_eval"] = held_out
        ho_mean = sum(h["success_rate"] for h in held_out) / len(held_out)
        mgr.log(f"Held-out success (n={args.eval_seeds}): {ho_mean*100:.1f}%")

    mgr.save_results(results)
    mgr.save_model(agent)
    if mgr.save_best_model(agent, avg_r):
        mgr.log("Saved as best-eval checkpoint.")
    mgr.save_curves(metrics.episodes)
    mgr.save_visitation(traj.trajectories, env)

    q_source = dict(agent.Q) if hasattr(agent, "Q") else agent
    try:
        mgr.save_policy_heatmap(q_source, env)
    except Exception as e:
        mgr.log(f"Policy heatmap skipped: {e}", "warning")

    try:
        mgr.save_rollout(agent, env)
    except Exception as e:
        mgr.log(f"Rollout viz skipped: {e}", "warning")

    # Replay WebP from one greedy episode starting at the canonical Start cell.
    # The renderer needs full-view frames (the static maze + agent position),
    # regardless of what the agent actually saw during the rollout.
    agent.set_deterministic(True)
    try:
        _, positions, _, total = simulate_episode(env, agent, args.max_steps,
                                                  at_start=True)
    finally:
        agent.set_deterministic(False)

    full_frames = [env.maze.copy() for _ in positions]
    sprites = _load_sprites(args)
    rm = RenderMaze(sprites)
    ReplayRecorder(rm).feed(full_frames, positions, None)
    mgr.save_replay(rm, fmt=args.replay_fmt,
                    sprite_size=args.sprite_size,
                    frame_skip=args.frame_skip,
                    max_frames=args.max_frames)
    mgr.log(f"Replay episode reward: {total:.3f}")

    try:
        mgr.save_html_report()
    except Exception as e:
        mgr.log(f"HTML report skipped: {e}", "warning")

    mgr.log(f"Run dir: {mgr.run_dir}")
    if web_thread is not None:
        mgr.log("Web viewer still running; Ctrl-C to exit.")
        try:
            web_thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    run(build_argparser().parse_args())
