"""Model bundle helpers shared by every training surface.

One place for the load/save/warm-start logic that used to be copy-pasted
across scripts/train_runpod.py, notebooks/train_agent.ipynb (cell 10) and
main.py — the copies had already drifted (main.py's resume skipped the
target-network sync).
"""

from __future__ import annotations

import pickle
from pathlib import Path


def module_of(agent):
    """The agent's torch module, or None for tabular Q."""
    return getattr(agent, "model", None) or getattr(agent, "ac", None)


def save_agent_model(agent, out_dir: Path, stem: str = "model") -> Path:
    """Write `<stem>.pt` (state_dict) or `<stem>.pkl` (tabular Q table).
    Returns the written path."""
    import torch
    out_dir = Path(out_dir)
    module = module_of(agent)
    if module is not None:
        path = out_dir / f"{stem}.pt"
        torch.save(module.state_dict(), path)
    else:
        path = out_dir / f"{stem}.pkl"
        path.write_bytes(pickle.dumps(dict(agent.Q)))
    return path


def warm_start(agent, path: str | Path) -> None:
    """Load saved weights into a freshly-created agent.

    Handles tabular Q (.pkl) and torch state-dicts (.pt); for value agents
    the target network is synced too — resuming with a random target net
    trains toward garbage for the first target_sync window."""
    import torch
    path = str(path)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            agent.Q.update(pickle.load(f))
        return
    sd = torch.load(path, map_location=getattr(agent, "device", "cpu"),
                    weights_only=True)
    module_of(agent).load_state_dict(sd)
    if hasattr(agent, "target_model"):
        agent.target_model.load_state_dict(sd)
