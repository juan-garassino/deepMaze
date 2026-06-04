"""Shared asset-bundle schema. No Prefect or MLflow imports so unit tests
can import this without the full flow runtime."""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = {
    "agent_type", "maze_width", "maze_height", "generator",
    "n_treasures", "n_lava", "seed", "run_name",
}


def validate_bundle(bundle: Path) -> None:
    cfg = json.loads((bundle / "config.json").read_text())
    missing = REQUIRED_KEYS - set(cfg)
    if missing:
        raise ValueError(f"config.json missing required keys: {missing}")
    if not any((bundle / f).exists() for f in ("model.pt", "model.pkl")):
        raise ValueError("bundle missing model.{pt,pkl}")
