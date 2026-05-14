"""Offline plotting — pure functions, save PNG, return path.

Consumes the same data as the EventBus does (lists of EpisodeEvent,
StepEvent sequences). Headless-friendly via Agg backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _plt():
    """Lazy import — matplotlib is ~3s to import, dominates nano runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------
# training curves
# --------------------------------------------------------------------------


def _ema(xs: Sequence[float], alpha: float = 0.1) -> list[float]:
    out: list[float] = []
    cur = None
    for x in xs:
        cur = x if cur is None else (1 - alpha) * cur + alpha * x
        out.append(cur)
    return out


def plot_training_curves(episodes: list[Any], out_path: str) -> str:
    plt = _plt()
    if not episodes:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no episode data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    rewards = [e.total_reward for e in episodes]
    lengths = [e.length for e in episodes]
    eps = [e.epsilon for e in episodes]
    losses = [e.loss for e in episodes if e.loss is not None]
    x = list(range(1, len(episodes) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    axes[0].plot(x, rewards, alpha=0.3, label="reward")
    axes[0].plot(x, _ema(rewards), label="ema", linewidth=2)
    axes[0].set_title("Episode reward"); axes[0].set_xlabel("episode")
    axes[0].legend(loc="lower right")

    axes[1].plot(x, lengths)
    axes[1].set_title("Episode length"); axes[1].set_xlabel("episode")

    if losses:
        lx = list(range(1, len(losses) + 1))
        axes[2].plot(lx, losses)
        axes[2].set_yscale("symlog")
    else:
        axes[2].text(0.5, 0.5, "no loss recorded\n(tabular Q)",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_axis_off()
    axes[2].set_title("Loss")

    axes[3].plot(x, eps)
    axes[3].set_title("Exploration (epsilon)"); axes[3].set_xlabel("episode")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# policy heatmap
# --------------------------------------------------------------------------


def plot_policy_heatmap(q_source, env, out_path: str) -> str:
    plt = _plt()
    """
    q_source can be:
      - a dict mapping flattened-state tuples -> action ndarray (tabular)
      - a callable: state_array -> q_values ndarray (DQN/PPO)
    For non-tabular agents we evaluate q_source on the observation produced by
    placing a single agent at each empty cell.
    """
    h, w = env.height, env.width
    V = np.full((h, w), np.nan)
    A = np.full((h, w), -1, dtype=int)

    if isinstance(q_source, dict) and q_source:
        # Tabular: pick a representative entry per cell by scanning the dict
        # for the agent-position cell value (4) over the maze grid.
        for i in range(h):
            for j in range(w):
                if env.maze[i, j] == 0:
                    continue
                obs = env.maze.copy()
                obs[i, j] = 5  # AGENT_BASE
                key = tuple(obs.flatten().tolist())
                if key in q_source:
                    qv = q_source[key]
                    V[i, j] = float(np.max(qv))
                    A[i, j] = int(np.argmax(qv))
    elif callable(q_source) or hasattr(q_source, "q_values_batch"):
        # Build one observation per non-wall cell, batch through the agent.
        cells = [(i, j) for i in range(h) for j in range(w)
                 if env.maze[i, j] != 0]
        obs_list = []
        for (i, j) in cells:
            o = env.maze.copy()
            o[i, j] = 5  # AGENT_BASE
            obs_list.append(o)
        obs_batch = np.stack(obs_list)
        if hasattr(q_source, "q_values_batch"):
            qs = q_source.q_values_batch(obs_batch)
        else:
            qs = np.stack([q_source(o) for o in obs_batch])
        for (i, j), qv in zip(cells, qs):
            V[i, j] = float(np.max(qv))
            A[i, j] = int(np.argmax(qv))

    fig, ax = plt.subplots(figsize=(0.5 * w + 2, 0.5 * h + 2))
    im = ax.imshow(V, cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="max Q (V)")

    # Walls as black overlay.
    walls = np.zeros((h, w, 4))
    walls[env.maze == 0] = [0, 0, 0, 1]
    ax.imshow(walls)

    # Goal/start markers (multi-treasure aware).
    sr, sc = env.start_pos
    ax.text(sc, sr, "S", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    for (tr, tc) in env.treasure_positions:
        ax.text(tc, tr, "T", ha="center", va="center",
                color="gold", fontsize=12, fontweight="bold")

    # Arrows for best action per cell.
    dyx = {0: (-0.35, 0), 1: (0, 0.35), 2: (0.35, 0), 3: (0, -0.35)}
    for i in range(h):
        for j in range(w):
            a = A[i, j]
            if a < 0:
                continue
            dy, dx = dyx[a]
            ax.arrow(j, i, dx, dy, head_width=0.18, head_length=0.18,
                     fc="white", ec="black", linewidth=0.6, length_includes_head=True)

    ax.set_title("Greedy policy + V(s)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# visitation heatmap
# --------------------------------------------------------------------------


def plot_visitation(trajectories: list[list[tuple[int, int]]], env, out_path: str) -> str:
    plt = _plt()
    h, w = env.height, env.width
    counts = np.zeros((h, w))
    for traj in trajectories:
        for pos in traj:
            counts[pos] += 1

    fig, ax = plt.subplots(figsize=(0.5 * w + 2, 0.5 * h + 2))
    masked = np.ma.masked_where(env.maze == 0, np.log1p(counts))
    im = ax.imshow(masked, cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1+visits)")
    sr, sc = env.start_pos
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontweight="bold")
    for (tr, tc) in env.treasure_positions:
        ax.text(tc, tr, "T", ha="center", va="center", color="gold", fontweight="bold")
    ax.set_title(f"Visitation ({len(trajectories)} eps)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# reward landscape (static maze figure)
# --------------------------------------------------------------------------


def plot_behavioral_rollout(agent, env, out_path: str,
                            max_rollout: int = 30) -> str:
    """For each non-wall cell, place the agent there and roll out
    deterministically; draw the trajectory trail + first-action arrow.

    Complements `plot_policy_heatmap`: that one shows what the agent
    *claims to know* (NaN for unvisited cells); this shows what the agent
    actually *does*. Honest about behaviour, no state-key lookup needed.
    """
    plt = _plt()
    from maze import HOLE, LAVA

    h, w = env.height, env.width
    arrows = np.full((h, w), -1, dtype=int)
    visits = np.zeros((h, w))

    prev = getattr(agent, "deterministic", False)
    agent.set_deterministic(True)
    try:
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if env.maze[i, j] in (HOLE, LAVA):
                    continue
                env.agent_positions = [(i, j)]
                if hasattr(agent, "on_episode_start"):
                    agent.on_episode_start()
                state = env.get_observation()
                first_action = None
                for _ in range(max_rollout):
                    a = agent.move(state)
                    if first_action is None:
                        first_action = a
                    ns, _, done, _ = env.step(a)
                    visits[env.agent_positions[0]] += 1
                    state = ns
                    if done:
                        break
                arrows[i, j] = first_action if first_action is not None else -1
    finally:
        agent.set_deterministic(prev)

    fig, ax = plt.subplots(figsize=(0.5 * w + 2, 0.5 * h + 2))
    masked = np.ma.masked_where(env.maze == HOLE, np.log1p(visits))
    im = ax.imshow(masked, cmap="cividis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log(1+rollout visits)")

    dyx = {0: (-0.35, 0), 1: (0, 0.35), 2: (0.35, 0), 3: (0, -0.35)}
    for i in range(h):
        for j in range(w):
            a = arrows[i, j]
            if a < 0:
                continue
            dy, dx = dyx[a]
            ax.arrow(j, i, dx, dy, head_width=0.22, head_length=0.22,
                     fc="white", ec="black", linewidth=0.6, length_includes_head=True)

    sr, sc = env.start_pos
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontweight="bold")
    for (tr, tc) in env.treasure_positions:
        ax.text(tc, tr, "T", ha="center", va="center", color="gold", fontweight="bold")
    # Mark lava cells
    for (i, j) in zip(*np.where(env.maze == LAVA)):
        ax.text(j, i, "X", ha="center", va="center", color="red", fontweight="bold")
    ax.set_title("Behavioral rollout (first action per cell)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_reward_landscape(env, out_path: str) -> str:
    plt = _plt()
    h, w = env.height, env.width
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[env.maze == 0] = (30, 30, 30)
    img[env.maze == 3] = (240, 200, 40)
    fig, ax = plt.subplots(figsize=(0.5 * w + 2, 0.5 * h + 2))
    ax.imshow(img)
    sr, sc = env.start_pos
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Maze layout")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
