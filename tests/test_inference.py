"""Inference pipeline: train a tiny Q-agent, save it, load it via the
inference helpers, stream a greedy episode, assert events were emitted.

Skips the FastAPI TestClient path on env mismatch but always runs the
pure-Python pipeline test.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from manager import MazeManager
from maze import MazeEnvironment
from seeding import seed_everything
from train import create_agent, simulate_episode_streaming, train_agent
from viz_events import EpisodeEvent, EventBus, RunEvent, StepEvent

# Ensure web/ is importable for helper functions.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "web"))


def test_inference_pipeline_q_agent(tmp_path, monkeypatch):
    """End-to-end: tiny train → save → load → stream → event count."""
    seed_everything(0)
    monkeypatch.chdir(tmp_path)

    # 1) Train a tiny Q-agent on a 5×5 open maze.
    env = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open")
    agent = create_agent("q", env, learning_rate=0.5)
    train_agent(env, agent, num_episodes=10, max_steps=20)

    # 2) Build a run dir matching the layout the inference loader expects.
    mgr = MazeManager(run_id="invtest")
    cfg = {
        "agent_type": "q",
        "maze_width": 5, "maze_height": 5,
        "density": 0.0, "generator": "open",
        "n_lava": 0, "n_treasures": 1, "partial": None,
        "seed": 0,
    }
    mgr.save_config(cfg)
    model_path = mgr.save_model(agent)
    assert model_path.exists()

    # 3) Use the inference helpers to reload it.
    from server import _find_model_file, _load_model_into  # noqa: E402
    found = _find_model_file(str(mgr.run_dir))
    assert found and Path(found).exists()

    env2 = MazeEnvironment(5, 5, density=0.0, seed=0, generator="open")
    agent2 = create_agent("q", env2)
    _load_model_into(agent2, found)
    agent2.set_deterministic(True)

    # 4) Stream one greedy episode through the bus; collect events.
    bus = EventBus()
    seen = []
    bus.subscribe(seen.append)
    simulate_episode_streaming(env2, agent2, bus, episode=0,
                               max_steps=20, at_start=True)
    bus.publish(RunEvent(kind="end"))

    step_events = [e for e in seen if isinstance(e, StepEvent)]
    ep_events = [e for e in seen if isinstance(e, EpisodeEvent)]
    end_events = [e for e in seen if isinstance(e, RunEvent) and e.kind == "end"]
    assert len(step_events) >= 1
    assert len(ep_events) == 1
    assert len(end_events) == 1


def test_find_model_prefers_best_then_final(tmp_path):
    from server import _find_model_file
    # Empty dir -> None
    assert _find_model_file(str(tmp_path)) is None
    # Only final
    (tmp_path / "model.pt").write_bytes(b"x")
    assert _find_model_file(str(tmp_path)).endswith("model.pt")
    # Best takes priority
    (tmp_path / "model.best.pt").write_bytes(b"y")
    assert _find_model_file(str(tmp_path)).endswith("model.best.pt")


# Optional FastAPI integration — skipped on env mismatch.
try:
    from fastapi.testclient import TestClient
    from server import create_app
    create_app()
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/starlette env mismatch")
def test_models_endpoint_lists_runs(tmp_path, monkeypatch):
    """When a valid run dir exists, /api/models surfaces it."""
    monkeypatch.chdir(tmp_path)
    # Build a minimal valid run dir.
    run_dir = tmp_path / "maze_rl_runs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"agent_type": "q"}))
    (run_dir / "model.pkl").write_bytes(b"placeholder")
    client = TestClient(create_app())
    r = client.get("/api/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert "demo" in names
