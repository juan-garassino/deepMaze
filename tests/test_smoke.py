"""End-to-end smoke: train q-agent for a handful of episodes, render replay."""

import numpy as np

from maze import MazeEnvironment, RenderMaze
from train import create_agent, train_agent, evaluate_agent, simulate_episode
from viz_events import EventBus
from recorders import MetricsCollector, TrajectoryCollector


def test_q_agent_smoke(tmp_path):
    env = MazeEnvironment(6, 6, density=0.1, seed=0)
    agent = create_agent("q", env, learning_rate=0.5, discount_factor=0.95)
    bus = EventBus()
    metrics = MetricsCollector()
    traj = TrajectoryCollector()
    bus.subscribe(metrics)
    bus.subscribe(traj)

    train_agent(env, agent, num_episodes=20, max_steps=80, bus=bus,
                policy_snapshot_every=10)

    assert len(metrics.episodes) == 20
    avg_r, avg_l, success = evaluate_agent(env, agent, num_episodes=5, max_steps=80)
    assert isinstance(avg_r, float)

    # Replay
    rm = RenderMaze(RenderMaze.placeholder_sprites(16))
    states, positions, _, _ = simulate_episode(env, agent, max_steps=80)
    for s, p in zip(states, positions):
        rm.add(s, p)
    out = tmp_path / "r.webp"
    rm.save(str(out), fmt="webp", sprite_size=16)
    assert out.exists() and out.stat().st_size > 0
