"""Unit test for the highest-leverage flow seam: bundle validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "drqn_test"
    bundle_dir.mkdir()
    (bundle_dir / "model.pt").write_bytes(b"dummy weights")
    return bundle_dir


def _load_validate():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from flows.bundle_schema import REQUIRED_KEYS, validate_bundle
    return validate_bundle, REQUIRED_KEYS


def _complete_cfg(name: str) -> dict:
    return {
        "agent_type": "drqn", "maze_width": 8, "maze_height": 8,
        "generator": "dfs", "n_treasures": 1, "n_lava": 0,
        "seed": 0, "run_name": name,
    }


def test_validate_accepts_complete_bundle(bundle: Path):
    validate, required = _load_validate()
    (bundle / "config.json").write_text(json.dumps(_complete_cfg(bundle.name)))
    validate(bundle)


@pytest.mark.parametrize("missing", [
    "agent_type", "maze_width", "maze_height", "generator",
    "n_treasures", "n_lava", "seed", "run_name",
])
def test_validate_rejects_missing_required_key(bundle: Path, missing: str):
    validate, _ = _load_validate()
    cfg = _complete_cfg(bundle.name)
    cfg.pop(missing)
    (bundle / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="missing required keys"):
        validate(bundle)


def test_validate_rejects_missing_model_file(tmp_path: Path):
    validate, _ = _load_validate()
    bundle = tmp_path / "no_model"
    bundle.mkdir()
    (bundle / "config.json").write_text(json.dumps(_complete_cfg("no_model")))
    with pytest.raises(ValueError, match="missing model"):
        validate(bundle)




def test_validate_accepts_pkl_model(tmp_path: Path):
    validate, _ = _load_validate()
    bundle = tmp_path / "tabular"
    bundle.mkdir()
    (bundle / "model.pkl").write_bytes(b"tabular q dict")
    (bundle / "config.json").write_text(json.dumps(_complete_cfg("tabular")))
    validate(bundle)
