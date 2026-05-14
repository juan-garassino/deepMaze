"""Same seed → same trajectory + same eval."""

import pytest
from maze import MazeEnvironment
from seeding import seed_everything
from train import create_agent, evaluate_agent, simulate_episode, train_agent

_TINY = {"q": {"learning_rate": 0.5},
         "dqn": {"batch_size": 16, "buffer_capacity": 128},
         "ppo": {"n_steps": 32, "minibatches": 2, "epochs": 2}}


def _run(agent_type, seed):
    seed_everything(seed)
    env = MazeEnvironment(5, 5, density=0.0, seed=seed, generator="open")
    agent = create_agent(agent_type, env, **_TINY[agent_type])
    train_agent(env, agent, num_episodes=5, max_steps=20)
    return evaluate_agent(env, agent, num_episodes=2, max_steps=20)


@pytest.mark.parametrize("agent_type", ["q", "dqn", "ppo"])
def test_seed_reproducibility(agent_type):
    a = _run(agent_type, seed=7)
    b = _run(agent_type, seed=7)
    assert a == b, f"{agent_type}: {a} != {b}"


def test_simulate_at_start_starts_at_start_pos():
    seed_everything(0)
    env = MazeEnvironment(8, 8, density=0.2, seed=0, generator="dfs")
    agent = create_agent("q", env)
    train_agent(env, agent, num_episodes=4, max_steps=40)
    _, positions, _, _ = simulate_episode(env, agent, max_steps=20, at_start=True)
    assert positions[0] == env.start_pos
