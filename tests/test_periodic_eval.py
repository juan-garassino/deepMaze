"""C7: periodic greedy eval inside train_agent — cadence, EvalEvents,
no buffer/epsilon pollution — and monotonic save_best_model."""

from __future__ import annotations

import pytest
from manager import MazeManager
from maze import MazeEnvironment
from train import create_agent, train_agent
from viz_events import EvalEvent, EventBus

_TINY = dict(width=5, height=5, generator="open", seed=0)


def test_eval_fires_at_cadence_and_emits_events():
    env = MazeEnvironment(**_TINY)
    agent = create_agent("q", env)
    bus = EventBus()
    eval_events = []
    bus.subscribe(lambda ev: eval_events.append(ev)
                  if isinstance(ev, EvalEvent) else None)
    calls = []
    train_agent(env, agent, num_episodes=4, max_steps=8, bus=bus,
                eval_every=2, eval_episodes=2,
                on_eval=lambda ep, m: calls.append((ep, m)))
    assert [ep for ep, _ in calls] == [1, 3]
    assert len(eval_events) == 2
    for _, m in calls:
        assert set(m) == {"mean_reward", "mean_length", "success_rate",
                          "revisit_rate"}
    ev = eval_events[0]
    assert ev.to_json()["type"] == "eval"


def test_eval_does_not_pollute_buffers():
    env = MazeEnvironment(**_TINY)
    dqn = create_agent("dqn", env, batch_size=4, buffer_capacity=512)
    train_agent(env, dqn, num_episodes=4, max_steps=5,
                eval_every=2, eval_episodes=3)
    assert len(dqn.memory) == 4 * 5  # only training steps stored

    env2 = MazeEnvironment(**_TINY, partial_view=1)
    drqn = create_agent("drqn", env2, batch_size=2, seq_len=4, burn_in=1,
                        buffer_capacity=64, lstm_hidden=16, enc_dim=16,
                        action_emb_dim=4)
    train_agent(env2, drqn, num_episodes=4, max_steps=5,
                eval_every=2, eval_episodes=3)
    assert len(drqn.buf) == 4  # eval episodes never reach the buffer


def test_epsilon_unchanged_by_periodic_eval():
    env = MazeEnvironment(**_TINY)
    a = create_agent("q", env)
    b = create_agent("q", env)
    train_agent(env, a, num_episodes=4, max_steps=5)
    train_agent(env, b, num_episodes=4, max_steps=5,
                eval_every=1, eval_episodes=2)
    assert a.epsilon == b.epsilon == pytest.approx(0.995 ** 4)
    assert not b.deterministic


def test_save_best_model_monotonic(tmp_path):
    mgr = MazeManager(base_dir=str(tmp_path))
    env = MazeEnvironment(**_TINY)
    agent = create_agent("q", env)
    assert mgr.save_best_model(agent, 0.2, episode=10) is True
    assert mgr.save_best_model(agent, 0.1, episode=20) is False
    assert mgr.save_best_model(agent, 0.5, episode=30) is True
    import json as _json
    payload = _json.loads((mgr.run_dir / "best_eval.json").read_text())
    assert payload == {"metric": 0.5, "metric_name": "eval_success_rate",
                       "episode": 30}


def test_save_best_model_reads_legacy_payload(tmp_path):
    mgr = MazeManager(base_dir=str(tmp_path))
    env = MazeEnvironment(**_TINY)
    agent = create_agent("q", env)
    (mgr.run_dir / "best_eval.json").write_text('{"eval_reward": 0.9}')
    assert mgr.save_best_model(agent, 0.5) is False
    assert mgr.save_best_model(agent, 0.95) is True
