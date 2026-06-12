"""R3: training/session.py — the shared train→eval→bundle cycle behind the
notebook and RunPod surfaces. Nano-scale, no MLflow."""

from __future__ import annotations

import json

from session import train_session


def test_train_session_full_cycle(tmp_path):
    res = train_session(
        agent_type="q", run_name="sess_q",
        env_kw=dict(width=5, height=5, generator="open", seed=0),
        num_episodes=8, max_steps=10,
        assets_dir=tmp_path / "assets", showcase_dir=tmp_path / "showcase",
        eval_episodes=2, eval_regenerate=False, eval_every=4,
        periodic_eval_episodes=2,
        seed=0, print_every=4, showcase_every=0,
        log_mlflow=False, mode_label="test",
    )
    out = tmp_path / "assets" / "sess_q"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["agent_type"] == "q"
    assert cfg["maze_width"] == 5 and cfg["partial"] is None
    assert "agent_hp" in cfg and "revisit_rate" not in cfg
    assert (out / "model.pkl").exists()  # tabular Q export
    assert (out / "viz" / "replay.webp").exists()
    assert res["model_path"].endswith("model.pkl")
    assert 0.0 <= res["eval_success_rate"] <= 1.0
    assert "eval_revisit_rate" in res


def test_train_session_warm_start_and_best(tmp_path):
    kw = dict(
        agent_type="dqn", run_name="sess_dqn",
        env_kw=dict(width=5, height=5, generator="open", seed=0),
        num_episodes=6, max_steps=8,
        assets_dir=tmp_path / "assets", showcase_dir=tmp_path / "showcase",
        agent_overrides=dict(batch_size=4, buffer_capacity=64),
        eval_episodes=2, eval_regenerate=False, eval_every=2,
        periodic_eval_episodes=2,
        seed=0, print_every=3, showcase_every=0,
        log_mlflow=False,
    )
    first = train_session(**kw)
    assert first["model_path"].endswith("model.pt")
    # periodic eval ran → a best snapshot exists (success may be 0 but the
    # first eval always improves on -1)
    assert first["best_model_path"] is not None

    second = train_session(**{**kw, "run_name": "sess_dqn2",
                              "warm_start_path": first["model_path"]})
    assert second["run_name"] == "sess_dqn2"
