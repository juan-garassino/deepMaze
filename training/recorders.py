"""EventBus subscribers: collect, render, stream.

Each recorder attaches to a bus once and is fed by training. No coupling
back to training code — adding a new recorder means one .subscribe() call.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from viz_events import EpisodeEvent, PolicyEvent, RunEvent, StepEvent


class MetricsCollector:
    """Buffers EpisodeEvents for later plotting."""

    def __init__(self):
        self.episodes: List[EpisodeEvent] = []

    def __call__(self, event):
        if isinstance(event, EpisodeEvent):
            self.episodes.append(event)


class TrajectoryCollector:
    """Buffers (episode -> list of positions) for visitation heatmap."""

    def __init__(self):
        self.trajectories: List[List[Tuple[int, int]]] = []
        self._current: List[Tuple[int, int]] = []
        self._ep: Optional[int] = None

    def __call__(self, event):
        if isinstance(event, StepEvent):
            if self._ep != event.episode:
                if self._current:
                    self.trajectories.append(self._current)
                self._current = []
                self._ep = event.episode
            self._current.append(tuple(event.position))
        elif isinstance(event, RunEvent) and event.kind == "end":
            if self._current:
                self.trajectories.append(self._current)
                self._current = []


class ReplayRecorder:
    """Drives RenderMaze.add() for one final-policy episode.

    Attach AFTER training: feed it states from a `simulate_episode` call.
    Kept stateless w.r.t. the bus for simplicity (used in main.py directly).
    """

    def __init__(self, render_maze):
        self.render_maze = render_maze

    def feed(self, states, positions, q_values_seq=None):
        for i, (s, p) in enumerate(zip(states, positions)):
            q = None if q_values_seq is None else q_values_seq[i]
            self.render_maze.add(s, p, q_values=q)


class TqdmTail:
    """Single-line live status on stderr without tqdm dependency."""

    def __init__(self, every: int = 1):
        self.every = every
        self._reward_ema = None

    def __call__(self, event):
        if isinstance(event, EpisodeEvent):
            self._reward_ema = (event.total_reward if self._reward_ema is None
                                else 0.9 * self._reward_ema + 0.1 * event.total_reward)
            if event.episode % self.every == 0:
                msg = (f"ep {event.episode:>5}  "
                       f"R={event.total_reward:+.3f}  ema={self._reward_ema:+.3f}  "
                       f"len={event.length:>4}  eps={event.epsilon:.3f}")
                if event.loss is not None:
                    msg += f"  loss={event.loss:.4f}"
                sys.stderr.write("\r" + msg + " " * 8)
                sys.stderr.flush()
        elif isinstance(event, RunEvent) and event.kind == "end":
            sys.stderr.write("\n")
            sys.stderr.flush()
