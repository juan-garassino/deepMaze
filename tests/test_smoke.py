"""Nano smoke: each agent runs end-to-end without errors.
Convergence is NOT asserted — these tests prove plumbing, not RL quality.
"""

import pytest
from maze import MazeEnvironment, RenderMaze
from recorders import MetricsCollector, TrajectoryCollector
from seeding import seed_everything
from train import create_agent, evaluate_agent, simulate_episode, train_agent
from viz_events import EventBus


@pytest.mark.parametrize("agent_type,episodes,extra", [
    ("q",   8, {"learning_rate": 0.5}),
    ("dqn", 6, {"batch_size": 16, "buffer_capacity": 256}),
    ("ppo", 6, {"n_steps": 32, "minibatches": 2, "epochs": 2}),
])
def test_agent_smoke(agent_type, episodes, extra, tmp_path):
    seed_everything(0)
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open")
    agent = create_agent(agent_type, env, **extra)
    bus = EventBus()
    metrics = MetricsCollector()
    bus.subscribe(metrics); bus.subscribe(TrajectoryCollector())

    train_agent(env, agent, num_episodes=episodes, max_steps=30, bus=bus,
                policy_snapshot_every=max(1, episodes))
    assert len(metrics.episodes) == episodes

    avg_r, avg_l, success = evaluate_agent(env, agent, num_episodes=3,
                                           max_steps=30)
    assert isinstance(avg_r, float)
    assert avg_l > 0

    rm = RenderMaze(RenderMaze.placeholder_sprites(16))
    states, positions, _, _ = simulate_episode(env, agent, max_steps=30,
                                               at_start=True)
    for s, p in zip(states, positions):
        rm.add(s, p)
    out = tmp_path / f"r-{agent_type}.webp"
    rm.save(str(out), fmt="webp", sprite_size=16)
    assert out.exists() and out.stat().st_size > 0
