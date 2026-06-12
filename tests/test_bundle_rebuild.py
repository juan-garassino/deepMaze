"""C8: bundles record effective hyperparameter overrides (agent_hp) +
env feature flags, and the rebuild path (create_agent + _load_model_into)
reproduces non-default architectures and aux-shaped observations."""

from __future__ import annotations

import os
import sys

import torch
from maze import MazeEnvironment
from train import create_agent

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "web"))

_TINY = dict(width=6, height=6, generator="open", seed=0, partial_view=2)


def _rebuild(cfg, model_path):
    from server import _load_model_into
    env = MazeEnvironment(
        width=cfg["maze_width"], height=cfg["maze_height"],
        generator=cfg["generator"], seed=cfg["seed"],
        partial_view=cfg["partial"],
        aux_features=cfg.get("aux_features", False),
        bump_penalty=cfg.get("bump_penalty", -0.1),
    )
    agent_kw = dict(cfg.get("agent_hp") or {})
    if cfg.get("net"):
        agent_kw["net"] = cfg["net"]
    agent = create_agent(cfg["agent_type"], env, **agent_kw)
    _load_model_into(agent, str(model_path))
    return env, agent


def test_rebuild_nondefault_arch(tmp_path):
    hp = dict(dim=16, heads=2, layers=1, max_ctx=8, seq_len=4, burn_in=1,
              batch_size=2, buffer_capacity=8)
    env = MazeEnvironment(**_TINY)
    agent = create_agent("dtqn", env, **hp)
    model_path = tmp_path / "model.pt"
    torch.save(agent.model.state_dict(), model_path)

    cfg = dict(agent_type="dtqn", net=None, maze_width=6, maze_height=6,
               generator="open", seed=0, partial=2, agent_hp=hp)
    _, agent2 = _rebuild(cfg, model_path)  # raises on shape mismatch
    sd1, sd2 = agent.model.state_dict(), agent2.model.state_dict()
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


def test_rebuild_without_agent_hp_fails_shape_check(tmp_path):
    """The pre-C8 failure mode: overrides not recorded → mismatch raises."""
    import pytest
    hp = dict(dim=16, heads=2, layers=1, max_ctx=8)
    env = MazeEnvironment(**_TINY)
    agent = create_agent("dtqn", env, **hp)
    model_path = tmp_path / "model.pt"
    torch.save(agent.model.state_dict(), model_path)
    cfg = dict(agent_type="dtqn", net=None, maze_width=6, maze_height=6,
               generator="open", seed=0, partial=2)  # no agent_hp
    with pytest.raises(RuntimeError):
        _rebuild(cfg, model_path)


def test_rebuild_respects_aux_flag(tmp_path):
    hp = dict(lstm_hidden=16, enc_dim=16, action_emb_dim=4,
              seq_len=4, burn_in=1, batch_size=2, buffer_capacity=8)
    env = MazeEnvironment(**_TINY, aux_features=True)
    agent = create_agent("drqn", env, **hp)
    model_path = tmp_path / "model.pt"
    torch.save(agent.model.state_dict(), model_path)

    cfg = dict(agent_type="drqn", net=None, maze_width=6, maze_height=6,
               generator="open", seed=0, partial=2,
               aux_features=True, agent_hp=hp)
    env2, agent2 = _rebuild(cfg, model_path)
    assert env2.aux_dim == 6
    assert "enc.aux_proj.weight" in agent2.model.state_dict()
    # greedy move works on the rebuilt agent with aux observations
    agent2.set_deterministic(True)
    agent2.on_episode_start()
    a = agent2.move(env2.reset(at_start=True))
    assert a in (0, 1, 2, 3)


def test_rebuild_tolerates_stale_agent_hp_keys(tmp_path):
    env = MazeEnvironment(**_TINY)
    agent = create_agent("dqn", env, batch_size=4, buffer_capacity=64)
    model_path = tmp_path / "model.pt"
    torch.save(agent.model.state_dict(), model_path)
    cfg = dict(agent_type="dqn", net=None, maze_width=6, maze_height=6,
               generator="open", seed=0, partial=2,
               agent_hp=dict(batch_size=4, buffer_capacity=64,
                             not_a_real_key=123))
    _rebuild(cfg, model_path)  # create_agent drops the unknown key


def test_bundles_warm_start_syncs_target(tmp_path):
    from bundles import save_agent_model, warm_start
    env = MazeEnvironment(**_TINY)
    src = create_agent("dqn", env, batch_size=4, buffer_capacity=32)
    path = save_agent_model(src, tmp_path)
    assert path.name == "model.pt"
    dst = create_agent("dqn", env, batch_size=4, buffer_capacity=32)
    warm_start(dst, path)
    sd_src, sd_tgt = src.model.state_dict(), dst.target_model.state_dict()
    assert all(torch.equal(sd_src[k], sd_tgt[k]) for k in sd_src)


def test_bundles_tabular_q_roundtrip(tmp_path):
    from bundles import save_agent_model, warm_start
    env = MazeEnvironment(**_TINY)
    src = create_agent("q", MazeEnvironment(width=6, height=6,
                                            generator="open", seed=0))
    obs = env.reset(at_start=True)
    src.update(obs, 1, 0.5, obs, False)
    path = save_agent_model(src, tmp_path)
    assert path.name == "model.pkl"
    dst = create_agent("q", env)
    warm_start(dst, path)
    assert len(dst.Q) == len(src.Q)
