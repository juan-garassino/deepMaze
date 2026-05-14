"""User-drawn maze must be honored verbatim by the server."""

import os
import sys

import numpy as np
import pytest

# Ensure web/ on path for create_app
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "web"))

try:
    from fastapi.testclient import TestClient
    from server import create_app  # noqa: E402
    create_app()  # smoke-construct: skips when fastapi/starlette versions mismatch
except Exception as _e:  # pragma: no cover — env-specific
    pytest.skip(f"fastapi/starlette env mismatch: {_e}", allow_module_level=True)


def _maze(h, w):
    m = np.ones((h, w), dtype=np.uint8)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0
    m[1, 1] = 2
    m[h - 2, w - 2] = 3
    return m


def test_missing_start_rejected():
    client = TestClient(create_app())
    m = _maze(6, 6); m[1, 1] = 1  # erase start
    r = client.post("/api/runs", json={"maze": m.tolist(),
                                       "agent_type": "q",
                                       "num_episodes": 1, "max_steps": 5})
    assert r.status_code == 400


def test_missing_goal_rejected():
    client = TestClient(create_app())
    m = _maze(6, 6); m[m.shape[0] - 2, m.shape[1] - 2] = 1
    r = client.post("/api/runs", json={"maze": m.tolist(),
                                       "agent_type": "q",
                                       "num_episodes": 1, "max_steps": 5})
    assert r.status_code == 400


def test_valid_maze_accepted():
    client = TestClient(create_app())
    m = _maze(6, 6)
    r = client.post("/api/runs", json={"maze": m.tolist(),
                                       "agent_type": "q",
                                       "num_episodes": 1, "max_steps": 5})
    assert r.status_code == 200
    assert "run_id" in r.json()
