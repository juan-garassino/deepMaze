"""C3: truncated kwarg on update(); PPO GAE carry zeroed at truncation;
actor-critic nets return logits (q_values stay softmaxed for viz)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from maze import MazeEnvironment
from ppo_agent import PPOAgent
from train import create_agent, train_agent

_TINY = dict(width=5, height=5, generator="open", seed=0)

_TINY_KW = {
    "q": {},
    "dqn": dict(batch_size=4, buffer_capacity=64),
    "ppo": dict(n_steps=8),
    "drqn": dict(batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                 lstm_hidden=16, enc_dim=16, action_emb_dim=4),
    "dtqn": dict(batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                 dim=16, heads=2, layers=1, max_ctx=8),
}


@pytest.mark.parametrize("agent_type", list(_TINY_KW))
def test_update_accepts_truncated_kwarg(agent_type):
    env = MazeEnvironment(**_TINY)
    agent = create_agent(agent_type, env, **_TINY_KW[agent_type])
    state = env.reset(at_start=True)
    if hasattr(agent, "on_episode_start"):
        agent.on_episode_start()
    a = agent.move(state)
    ns, r, d, _ = env.step(a)
    agent.update(state, a, r, ns, d, truncated=not d)


def test_gae_carry_zeroed_at_truncation():
    gamma, lam = 0.9, 0.8
    r = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = torch.zeros(4)
    values = torch.zeros(4)
    next_values = torch.tensor([0.5, 0.5, 0.5, 0.5])
    # Truncation after step 1: steps 0-1 are one episode, 2-3 the next.
    trunc = torch.tensor([0.0, 1.0, 0.0, 0.0])

    adv = PPOAgent._compute_gae(r, d, trunc, values, next_values, gamma, lam)

    # Hand-computed, backwards. delta_t = r + gamma*nv - v.
    d3 = 4.0 + gamma * 0.5
    d2 = 3.0 + gamma * 0.5
    d1 = 2.0 + gamma * 0.5
    d0 = 1.0 + gamma * 0.5
    a3 = d3
    a2 = d2 + gamma * lam * a3
    a1 = d1                      # carry zeroed: a1 must NOT include a2
    a0 = d0 + gamma * lam * a1
    expected = torch.tensor([a0, a1, a2, a3])
    assert torch.allclose(adv, expected, atol=1e-6)

    # Without the trunc flag the old (buggy) behavior leaks a2 into a1.
    adv_leaky = PPOAgent._compute_gae(r, d, torch.zeros(4), values,
                                      next_values, gamma, lam)
    assert not torch.allclose(adv_leaky[1], adv[1])


def test_gae_done_still_resets_carry_and_blocks_bootstrap():
    gamma, lam = 0.9, 0.8
    r = torch.tensor([1.0, 2.0])
    d = torch.tensor([1.0, 0.0])
    trunc = torch.zeros(2)
    values = torch.zeros(2)
    next_values = torch.tensor([10.0, 0.5])
    adv = PPOAgent._compute_gae(r, d, trunc, values, next_values, gamma, lam)
    # Terminal step 0: no bootstrap of next_values[0], no carry from step 1.
    assert adv[0] == pytest.approx(1.0)


def test_ppo_q_values_are_probs():
    env = MazeEnvironment(**_TINY)
    agent = create_agent("ppo", env)
    qv = agent.q_values(env.reset(at_start=True))
    assert qv.shape == (4,)
    assert np.all(qv >= 0)
    assert qv.sum() == pytest.approx(1.0, abs=1e-5)


def test_ppo_trains_through_truncations():
    env = MazeEnvironment(**_TINY)
    agent = create_agent("ppo", env, n_steps=8)
    train_agent(env, agent, num_episodes=4, max_steps=5)  # every ep truncates
    assert agent.last_loss is not None
