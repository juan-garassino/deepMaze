import numpy as np

from maze import MazeEnvironment
from train import create_agent, train_agent
from seeding import seed_everything


def test_ppo_deterministic_action_stable():
    seed_everything(0)
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open")
    agent = create_agent("ppo", env, n_steps=16, minibatches=2, epochs=2)
    train_agent(env, agent, num_episodes=3, max_steps=15)
    s = env.reset(at_start=True)
    agent.set_deterministic(True)
    acts = {agent.move(s) for _ in range(10)}
    assert len(acts) == 1, f"deterministic move should be stable, got {acts}"
    agent.set_deterministic(False)


def test_qdqn_set_deterministic_zeros_epsilon():
    seed_everything(0)
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open")
    for at in ("q", "dqn"):
        agent = create_agent(at, env)
        before = agent.epsilon
        agent.set_deterministic(True)
        assert agent.epsilon == 0.0
        agent.set_deterministic(False)
        assert agent.epsilon == before
