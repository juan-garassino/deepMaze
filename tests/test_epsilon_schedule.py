"""C2: epsilon decays once per EPISODE via on_episode_end(), never inside
update(). Pins the fix for the per-step-collapse bug."""

from __future__ import annotations

import numpy as np
import pytest
from dqn_agent import DQNAgent
from drqn_agent import DRQNAgent
from dtqn_agent import DTQNAgent
from maze import MazeEnvironment
from q_agent import QAgent
from train import create_agent, train_agent

_TINY = dict(width=5, height=5, generator="open", seed=0)


def _agents():
    env = MazeEnvironment(**_TINY)
    obs = env.get_observation()
    n = int(np.prod(obs.shape))
    return [
        QAgent(action_size=4),
        DQNAgent(state_size=n, action_size=4, batch_size=4, buffer_capacity=64),
        DRQNAgent(state_size=n, action_size=4, grid_shape=obs.shape,
                  batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                  lstm_hidden=16, enc_dim=16, action_emb_dim=4),
        DTQNAgent(state_size=n, action_size=4, grid_shape=obs.shape,
                  batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                  dim=16, heads=2, layers=1, max_ctx=8),
    ], env


def test_update_does_not_decay_epsilon():
    agents, env = _agents()
    state = env.reset(at_start=True)
    for agent in agents:
        if hasattr(agent, "on_episode_start"):
            agent.on_episode_start()
        eps0 = agent.epsilon
        for _ in range(20):
            a = agent.move(state)
            ns, r, d, _ = env.step(a)
            agent.update(state, a, r, ns, d)
            state = ns if not d else env.reset(at_start=True)
        assert agent.epsilon == eps0, type(agent).__name__


def test_on_episode_end_decays_once():
    agents, _ = _agents()
    for agent in agents:
        eps0 = agent.epsilon
        agent.on_episode_end()
        assert agent.epsilon == pytest.approx(eps0 * agent.epsilon_decay), \
            type(agent).__name__


def test_on_episode_end_respects_floor():
    agent = QAgent(action_size=4, exploration_decay=0.1, min_epsilon=0.3)
    for _ in range(10):
        agent.on_episode_end()
    assert agent.epsilon == 0.3


def test_on_episode_end_noop_while_deterministic():
    agent = QAgent(action_size=4)
    agent.set_deterministic(True)
    agent.on_episode_end()
    agent.set_deterministic(False)
    assert agent.epsilon == 1.0  # restored, undecayed


def test_train_agent_decays_per_episode():
    env = MazeEnvironment(**_TINY)
    agent = create_agent("q", env)
    train_agent(env, agent, num_episodes=3, max_steps=10, bus=None)
    assert agent.epsilon == pytest.approx(0.995 ** 3)


def test_create_agent_ppo_tolerates_exploration_decay():
    env = MazeEnvironment(**_TINY)
    agent = create_agent("ppo", env, exploration_decay=0.998,
                         buffer_capacity=500)
    assert type(agent).__name__ == "PPOAgent"
    agent.on_episode_end()  # inherited no-op (no epsilon_decay attr)
    assert agent.epsilon == 0.0
