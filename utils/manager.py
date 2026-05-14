"""MazeManager: single owner of per-run artifact paths.

Run folder layout:
    maze_rl_runs/run_YYYYMMDD_HHMMSS/
        config.json
        results.json
        maze_rl.log
        model.pt
        viz/
            replay.webp
            curves.png
            policy.png
            visitation.png
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MazeManager:
    def __init__(self, base_dir: str = "maze_rl_runs", run_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        ts = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = ts
        self.run_dir = self.base_dir / f"run_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "viz").mkdir(exist_ok=True)
        self.logger = self._setup_logger()

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    def viz_dir(self) -> Path:
        return self.run_dir / "viz"

    def path(self, *parts) -> Path:
        return self.run_dir.joinpath(*parts)

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"MazeRL.{self.run_id}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fh = logging.FileHandler(self.run_dir / "maze_rl.log")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
        logger.propagate = False
        return logger

    def log(self, msg: str, level: str = "info") -> None:
        getattr(self.logger, level, self.logger.info)(msg)

    def print_and_log(self, msg: str, level: str = "info") -> None:
        self.log(msg, level)

    # ------------------------------------------------------------------
    # artifacts
    # ------------------------------------------------------------------
    def save_config(self, config: Dict[str, Any]) -> Path:
        p = self.run_dir / "config.json"
        with open(p, "w") as f:
            json.dump({k: _jsonable(v) for k, v in config.items()}, f, indent=2)
        self.log(f"Config saved to {p}")
        return p

    def save_results(self, results: Dict[str, Any]) -> Path:
        p = self.run_dir / "results.json"
        with open(p, "w") as f:
            json.dump({k: _jsonable(v) for k, v in results.items()}, f, indent=2)
        self.log(f"Results saved to {p}")
        return p

    def save_model(self, agent, filename: str = "model") -> Path:
        if hasattr(agent, "Q"):  # tabular
            p = self.run_dir / f"{filename}.pkl"
            with open(p, "wb") as f:
                pickle.dump({k: v for k, v in agent.Q.items()}, f)
        else:
            import torch
            module = getattr(agent, "model", None) or getattr(agent, "ac", None)
            p = self.run_dir / f"{filename}.pt"
            torch.save(module.state_dict(), p)
        self.log(f"Model saved to {p}")
        return p

    # --- viz convenience -------------------------------------------------
    def save_replay(self, render_maze, fmt: str = "webp", **kw) -> Path:
        out = self.viz_dir() / f"replay.{fmt}"
        render_maze.save(str(out), fmt=fmt, **kw)
        self.log(f"Replay saved to {out}")
        return out

    def save_gif(self, gif_path: str) -> Path:
        """Back-compat: copy a pre-rendered GIF into viz/."""
        dst = self.viz_dir() / "replay.gif"
        shutil.copy(gif_path, dst)
        self.log(f"GIF copied to {dst}")
        return dst

    def save_curves(self, episodes: List[Any]) -> Path:
        from visualizations import plot_training_curves  # lazy
        out = self.viz_dir() / "curves.png"
        plot_training_curves(episodes, str(out))
        self.log(f"Curves saved to {out}")
        return out

    def save_rollout(self, agent, env, max_rollout: int = 30) -> Path:
        from visualizations import plot_behavioral_rollout
        out = self.viz_dir() / "rollout.png"
        plot_behavioral_rollout(agent, env, str(out), max_rollout=max_rollout)
        self.log(f"Rollout viz saved to {out}")
        return out

    def save_best_model(self, agent, eval_reward: float) -> bool:
        """Persist agent as model.best.* if eval_reward improves the running best.
        Returns True if persisted."""
        best_file = self.run_dir / "best_eval.json"
        prev = float("-inf")
        if best_file.exists():
            try:
                prev = float(json.loads(best_file.read_text())["eval_reward"])
            except Exception:
                pass
        if eval_reward <= prev:
            return False
        self.save_model(agent, filename="model.best")
        best_file.write_text(json.dumps({"eval_reward": eval_reward}))
        return True

    def save_policy_heatmap(self, q_source, env) -> Path:
        from visualizations import plot_policy_heatmap
        out = self.viz_dir() / "policy.png"
        plot_policy_heatmap(q_source, env, str(out))
        self.log(f"Policy heatmap saved to {out}")
        return out

    def save_visitation(self, trajectories: List[List], env) -> Path:
        from visualizations import plot_visitation
        out = self.viz_dir() / "visitation.png"
        plot_visitation(trajectories, env, str(out))
        self.log(f"Visitation saved to {out}")
        return out


def _jsonable(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return repr(v)
