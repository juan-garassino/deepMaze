"""C4: shared EpisodeBuffer — padding mask, idempotent end_episode, and
truncated/final episodes reaching the buffer via on_episode_end()."""

from __future__ import annotations

import numpy as np
import torch
from episode_buffer import NO_ACTION, EpisodeBuffer
from maze import MazeEnvironment
from train import create_agent, train_agent

_TINY = dict(width=5, height=5, generator="open", seed=0)
_DRQN_KW = dict(batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                lstm_hidden=16, enc_dim=16, action_emb_dim=4)
_DTQN_KW = dict(batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                dim=16, heads=2, layers=1, max_ctx=8)


def _obs(v):
    return np.full((3, 3), v, dtype=np.uint8)


def _fill_episode(buf, n_steps, done_last=True):
    buf.start_episode()
    for i in range(n_steps):
        buf.add_step(_obs(i), i % 4, 0.1 * i, _obs(i + 1),
                     done_last and i == n_steps - 1)


def test_sample_mask_marks_padding():
    buf = EpisodeBuffer(capacity=4)
    _fill_episode(buf, 3)
    batch = buf.sample(1, seq_len=8)
    assert batch["mask"].shape == (1, 8)
    np.testing.assert_array_equal(batch["mask"][0],
                                  [1, 1, 1, 0, 0, 0, 0, 0])
    # pads repeat the terminal transition
    np.testing.assert_array_equal(batch["obs"][0, 3], batch["obs"][0, 2])
    assert batch["done"][0, 2] == 1.0 and batch["done"][0, 7] == 1.0
    # full-length crops carry an all-ones mask
    _fill_episode(buf, 10)
    full = [m for m in buf.sample(2, seq_len=8)["mask"] if m.all()]
    assert full


def test_prev_action_chain():
    buf = EpisodeBuffer(capacity=2)
    _fill_episode(buf, 4)
    b = buf.sample(1, seq_len=4)
    assert b["prev_action"][0, 0] == NO_ACTION
    np.testing.assert_array_equal(b["prev_action"][0, 1:], b["action"][0, :-1])


def test_end_episode_flushes_truncated_and_is_idempotent():
    buf = EpisodeBuffer(capacity=4)
    _fill_episode(buf, 3, done_last=False)  # truncated — no done flag
    assert len(buf) == 0
    buf.end_episode()
    assert len(buf) == 1
    buf.end_episode()
    buf.end_episode()
    assert len(buf) == 1  # idempotent


def test_capacity_is_in_episodes():
    buf = EpisodeBuffer(capacity=2)
    for _ in range(5):
        _fill_episode(buf, 2)
    assert len(buf) == 2


def test_final_episode_reaches_buffer_via_train_agent():
    # max_steps=3 on a 5x5: every episode truncates; without the
    # on_episode_end flush none of them would ever be sampled.
    for agent_type, kw in (("drqn", _DRQN_KW), ("dtqn", _DTQN_KW)):
        env = MazeEnvironment(**_TINY)
        agent = create_agent(agent_type, env, **kw)
        train_agent(env, agent, num_episodes=4, max_steps=3)
        assert len(agent.buf) == 4, agent_type


def test_dtqn_forward_finite_with_key_padding_mask():
    env = MazeEnvironment(**_TINY, partial_view=1)
    agent = create_agent("dtqn", env, **_DTQN_KW)
    obs = env.reset(at_start=True)
    B, T = 2, 4
    obs_t = torch.from_numpy(np.stack([[obs] * T] * B)).long()
    pa = torch.full((B, T), -1, dtype=torch.long)
    kpm = torch.tensor([[False, False, True, True],
                        [False, False, False, False]])
    q = agent.model(obs_t.to(agent.device), pa.to(agent.device),
                    key_padding_mask=kpm.to(agent.device))
    assert torch.isfinite(q).all()


def test_memory_agents_learn_with_masked_loss():
    for agent_type, kw in (("drqn", _DRQN_KW), ("dtqn", _DTQN_KW)):
        env = MazeEnvironment(**_TINY, partial_view=1)
        agent = create_agent(agent_type, env, **kw)
        train_agent(env, agent, num_episodes=6, max_steps=6)
        assert agent.last_loss is not None and np.isfinite(agent.last_loss), \
            agent_type


def test_learn_every_gates_gradient_steps():
    env = MazeEnvironment(**_TINY)
    kw = dict(_DRQN_KW, learn_every=1000)  # never reached in 20 steps
    agent = create_agent("drqn", env, **kw)
    train_agent(env, agent, num_episodes=4, max_steps=5)
    assert agent.last_loss is None
    kw = dict(_DRQN_KW, learn_every=1)
    agent = create_agent("drqn", env, **kw)
    train_agent(env, agent, num_episodes=4, max_steps=5)
    assert agent.last_loss is not None


def test_stored_hidden_state_sampled_at_chunk_start():
    env = MazeEnvironment(**_TINY, partial_view=1)
    agent = create_agent("drqn", env, **_DRQN_KW)
    train_agent(env, agent, num_episodes=4, max_steps=6)
    batch = agent.buf.sample(2, seq_len=4)
    ie = batch["init_extra"]
    assert ie is not None
    layers = agent.model.lstm.num_layers
    hsize = agent.model.lstm.hidden_size
    assert ie.shape == (2, 2, layers, hsize)  # (B, h/c, layers, H)
    assert np.isfinite(ie).all()


def test_buffer_without_extras_returns_none():
    buf = EpisodeBuffer(capacity=2)
    _fill_episode(buf, 3)
    assert buf.sample(1, seq_len=4)["init_extra"] is None


def test_drqn_learns_from_stored_state_burn_in():
    env = MazeEnvironment(**_TINY, partial_view=1)
    agent = create_agent("drqn", env, **_DRQN_KW)
    train_agent(env, agent, num_episodes=6, max_steps=6)
    assert agent.last_loss is not None and np.isfinite(agent.last_loss)
