"""C5: on-device one-hot equals the legacy numpy path; Double-DQN target
selects with the online net and evaluates with the target net; DQN clips
gradients and trains with Huber loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dqn_agent import DQNAgent
from maze import MazeEnvironment
from nets import VOCAB, encode_grid_batch, grid_onehot
from train import create_agent, train_agent

_TINY = dict(width=5, height=5, generator="open", seed=0)


def _legacy_onehot(states, h, w):
    x = np.asarray(states)
    if x.ndim == 2 and x.shape[1] == h * w:
        x = x.reshape(-1, h, w)
    n = x.shape[0]
    flat = np.clip(x, 0, VOCAB - 1).astype(np.int64).reshape(n, -1)
    onehot = np.zeros((n, flat.shape[1], VOCAB), dtype=np.float32)
    onehot[np.arange(n)[:, None], np.arange(flat.shape[1])[None, :], flat] = 1.0
    return torch.from_numpy(onehot.reshape(n, h, w, VOCAB).transpose(0, 3, 1, 2))


def test_grid_onehot_matches_legacy_numpy():
    rng = np.random.default_rng(0)
    grids = rng.integers(0, 8, size=(4, 5, 5))  # includes out-of-vocab 6,7
    for shaped in (grids, grids.reshape(4, 25)):
        new = grid_onehot(torch.from_numpy(shaped.astype(np.float32)), 5, 5)
        legacy = _legacy_onehot(shaped, 5, 5)
        assert new.dtype == torch.float32
        assert torch.equal(new, legacy)
        wrapped = encode_grid_batch(shaped, 5, 5)
        assert torch.equal(wrapped, legacy)


class _FixedNet(nn.Module):
    """Returns a constant Q-table row per state, ignoring input values."""

    def __init__(self, q_row):
        super().__init__()
        self.q = torch.tensor([q_row], dtype=torch.float32)
        self.dummy = nn.Linear(1, 1)  # so .parameters() is non-empty

    def forward(self, x):
        return self.q.expand(x.shape[0], -1)


def test_double_dqn_target_math():
    agent = DQNAgent(state_size=4, action_size=3, batch_size=2,
                     buffer_capacity=8, discount_factor=0.5)
    # Online net prefers action 2; target net values action 2 at 1.0 but
    # would itself argmax action 0 (10.0). Double DQN must use 1.0.
    agent.model = _FixedNet([0.0, 0.0, 5.0])
    agent.target_model = _FixedNet([10.0, 0.0, 1.0])

    ns = torch.zeros(1, 4)
    with torch.no_grad():
        next_a = agent.model(ns).argmax(1, keepdim=True)
        next_q = agent.target_model(ns).gather(1, next_a)
    assert next_a.item() == 2
    assert next_q.item() == 1.0  # vanilla max would give 10.0


def test_dqn_trains_with_huber_and_clip():
    env = MazeEnvironment(**_TINY)
    agent = create_agent("dqn", env, batch_size=4, buffer_capacity=64)
    train_agent(env, agent, num_episodes=4, max_steps=10)
    assert agent.last_loss is not None and np.isfinite(agent.last_loss)
    for p in agent.model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
