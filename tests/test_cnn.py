"""CNN backbone runs end-to-end on full-view and partial-view obs."""

import pytest
from maze import MazeEnvironment
from seeding import seed_everything
from train import create_agent, simulate_episode, train_agent


@pytest.mark.parametrize("agent_type", ["dqn", "ppo"])
def test_cnn_runs_full_view(agent_type):
    seed_everything(0)
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open")
    kw = {"net": "cnn"}
    if agent_type == "ppo":
        kw.update(n_steps=16, minibatches=2, epochs=2)
    if agent_type == "dqn":
        kw.update(batch_size=8, buffer_capacity=64)
    agent = create_agent(agent_type, env, **kw)
    train_agent(env, agent, num_episodes=3, max_steps=15)
    states, positions, _, _ = simulate_episode(env, agent, max_steps=15,
                                               at_start=True)
    assert len(states) >= 2


@pytest.mark.parametrize("agent_type", ["dqn", "ppo"])
def test_cnn_runs_partial_view(agent_type):
    seed_everything(0)
    env = MazeEnvironment(6, 6, density=0.0, seed=0, generator="open",
                          partial_view=1)
    kw = {"net": "cnn"}
    if agent_type == "ppo":
        kw.update(n_steps=16, minibatches=2, epochs=2)
    if agent_type == "dqn":
        kw.update(batch_size=8, buffer_capacity=64)
    agent = create_agent(agent_type, env, **kw)
    train_agent(env, agent, num_episodes=3, max_steps=15)
