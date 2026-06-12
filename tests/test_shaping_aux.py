"""C6: opt-in reward shaping (potential-based, telescoping) and aux
observation features. Defaults-off behavior must be byte-identical."""

from __future__ import annotations

import numpy as np
import pytest
from maze import AUX_DIM, EXIT, MazeEnvironment
from train import create_agent, train_agent

_TINY = dict(width=6, height=6, generator="open", seed=0)
_DRQN_KW = dict(batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                lstm_hidden=16, enc_dim=16, action_emb_dim=4)
_DTQN_KW = dict(batch_size=2, seq_len=4, burn_in=1, buffer_capacity=8,
                dim=16, heads=2, layers=1, max_ctx=8)


def test_defaults_off_unchanged():
    env = MazeEnvironment(**_TINY)
    obs = env.reset(at_start=True)
    assert obs.ndim == 2 and obs.shape == (6, 6)
    assert env.aux_dim == 0
    _, r, d, _ = env.step(0)  # bump into top wall from (1,1)
    assert r == -0.1 and not d
    _, r, d, _ = env.step(2)  # step onto LAND
    assert r == -0.01 and not d
    grid, aux = env.split_observation(env.get_observation())
    assert grid.shape == (6, 6) and aux is None


def test_aux_observation_shape_and_bounds():
    env = MazeEnvironment(**_TINY, aux_features=True)
    obs = env.reset(at_start=True)
    assert obs.ndim == 1 and obs.shape == (36 + AUX_DIM,)
    assert obs.dtype == np.float32
    grid, aux = env.split_observation(obs)
    assert grid.shape == (6, 6)
    np.testing.assert_array_equal(grid, MazeEnvironment(**_TINY).reset(at_start=True))
    assert aux.shape == (AUX_DIM,)
    r, c = env.agent_positions[0]
    assert aux[0] == pytest.approx(r / 5) and aux[1] == pytest.approx(c / 5)
    assert -1.0 <= aux[2] <= 1.0 and -1.0 <= aux[3] <= 1.0  # unit vector
    assert 0.0 <= aux[4] <= 1.0  # normalized distance
    assert aux[5] == 1.0  # remaining fraction (non-collect_all)


def test_aux_direction_points_at_treasure():
    env = MazeEnvironment(**_TINY, aux_features=True)
    env.reset(at_start=True)
    _, aux = env.split_observation(env.get_observation())
    # start (1,1), treasure (4,4): direction is down-right
    assert aux[2] > 0 and aux[3] > 0


def test_remaining_fraction_in_collect_all():
    env = MazeEnvironment(width=7, height=7, generator="open", seed=0,
                          n_treasures=2, collect_all=True, aux_features=True)
    env.reset(at_start=True)
    _, aux0 = env.split_observation(env.get_observation())
    assert aux0[5] == 1.0
    tr = sorted(env.treasure_positions)[0]
    # walk to the first treasure
    r, c = env.start_pos
    while r < tr[0]:
        env.step(2); r += 1
    while c < tr[1]:
        env.step(1); c += 1
    _, aux1 = env.split_observation(env.get_observation())
    assert aux1[5] == pytest.approx(0.5)


def test_shaping_telescopes_on_greedy_path():
    base = MazeEnvironment(**_TINY)
    shaped = MazeEnvironment(**_TINY, reward_shaping=True,
                             shaping_gamma=1.0, shaping_coef=0.05)
    for env in (base, shaped):
        env.reset(at_start=True)
    tr = base.treasure_positions[0]
    assert base.maze[tr] == EXIT
    actions = [2] * (tr[0] - 1) + [1] * (tr[1] - 1)
    phi_s0 = shaped._phi(shaped.start_pos)
    tot_base = tot_shaped = 0.0
    for a in actions:
        _, rb, db, _ = base.step(a)
        _, rs, ds, _ = shaped.step(a)
        tot_base += rb
        tot_shaped += rs
    assert db and ds
    # with gamma=1, sum of shaping terms telescopes to Φ(terminal)-Φ(s0)=-Φ(s0)
    assert tot_shaped - tot_base == pytest.approx(-phi_s0, abs=1e-5)
    assert tot_shaped > tot_base  # moving toward the treasure is rewarded


def test_shaping_rewards_approach_and_penalizes_retreat():
    env = MazeEnvironment(**_TINY, reward_shaping=True, shaping_coef=0.05)
    env.reset(at_start=True)
    _, r_toward, _, _ = env.step(2)   # toward treasure at (4,4)
    assert r_toward > -0.01           # base step penalty offset by shaping
    _, r_away, _, _ = env.step(0)     # back up
    assert r_away < -0.01


def test_create_agent_rejects_q_with_aux():
    env = MazeEnvironment(**_TINY, aux_features=True)
    with pytest.raises(ValueError, match="aux"):
        create_agent("q", env)


@pytest.mark.parametrize("agent_type,kw", [
    ("dqn", dict(net="cnn", batch_size=4, buffer_capacity=64)),
    ("dqn", dict(net="mlp", batch_size=4, buffer_capacity=64)),
    ("ppo", dict(net="cnn", n_steps=8)),
    ("drqn", _DRQN_KW),
    ("dtqn", _DTQN_KW),
])
def test_aux_agents_train_smoke(agent_type, kw):
    env = MazeEnvironment(**_TINY, partial_view=2, aux_features=True,
                          reward_shaping=True)
    agent = create_agent(agent_type, env, **kw)
    train_agent(env, agent, num_episodes=3, max_steps=8)
    qv = agent.q_values(env.reset(at_start=True))
    assert qv.shape == (4,)
    assert np.isfinite(qv).all()


def test_warm_start_keys_stable_across_sizes():
    # same partial window → identical state-dict keys/shapes across maze sizes
    kw = dict(partial_view=2, aux_features=True)
    env_small = MazeEnvironment(width=8, height=8, generator="open", seed=0, **kw)
    env_big = MazeEnvironment(width=12, height=16, generator="open", seed=0, **kw)
    a = create_agent("drqn", env_small, **_DRQN_KW)
    b = create_agent("drqn", env_big, **_DRQN_KW)
    sd_a, sd_b = a.model.state_dict(), b.model.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert sd_a[k].shape == sd_b[k].shape, k
    b.model.load_state_dict(sd_a)  # transfer must not raise
