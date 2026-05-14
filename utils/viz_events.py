"""Typed event bus for visualization consumers.

Training emits events; subscribers (file recorders, tqdm tail, SSE hub)
react. Subscribing == one function call. Adding a new viz surface requires
no changes to training code.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class StepEvent:
    episode: int
    step: int
    state: np.ndarray
    position: tuple[int, int]
    action: int
    reward: float
    done: bool
    q_values: np.ndarray | None = None
    memory: dict | None = None  # {'kind': 'lstm_hidden'|'attention_row', 'data': [...]}

    def to_json(self) -> dict:
        """Backwards-compat: full payload. Prefer to_json_delta for live SSE."""
        return self.to_json_full()

    def to_json_full(self) -> dict:
        return {
            "type": "step",
            "episode": self.episode,
            "step": self.step,
            "state": self.state.tolist(),
            "position": list(self.position),
            "action": int(self.action),
            "reward": float(self.reward),
            "done": bool(self.done),
            "q_values": None if self.q_values is None else self.q_values.tolist(),
            "memory": self.memory,
        }

    def to_json_delta(self) -> dict:
        return {
            "type": "step_delta",
            "episode": self.episode,
            "step": self.step,
            "position": list(self.position),
            "action": int(self.action),
            "reward": float(self.reward),
            "done": bool(self.done),
            "q_values": None if self.q_values is None else self.q_values.tolist(),
            "memory": self.memory,
        }


@dataclass
class EpisodeEvent:
    episode: int
    total_reward: float
    length: int
    epsilon: float
    loss: float | None = None
    success: bool = False

    def to_json(self) -> dict:
        return {"type": "episode", **asdict(self)}


@dataclass
class PolicyEvent:
    episode: int
    snapshot: Any = field(repr=False)

    def to_json(self) -> dict:
        # snapshot is intentionally not serialized over SSE (too large).
        return {"type": "policy", "episode": self.episode}


@dataclass
class RunEvent:
    kind: str          # 'start' | 'end'
    info: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"type": "run", "kind": self.kind, "info": self.info}


VizEvent = Any  # StepEvent | EpisodeEvent | PolicyEvent | RunEvent


class EventBus:
    """Synchronous publish + opt-in queue-based subscribers.

    `subscribe(fn)` for fast in-process handlers (called on the publishing
    thread). `subscribe_queue(maxsize)` returns a `queue.Queue` for handlers
    that need to drain on a different thread (e.g. SSE).
    """

    def __init__(self) -> None:
        self._handlers: list[Callable[[VizEvent], None]] = []
        self._queues: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[VizEvent], None]) -> Callable[[], None]:
        with self._lock:
            self._handlers.append(fn)
        return lambda: self._handlers.remove(fn) if fn in self._handlers else None

    def subscribe_queue(self, maxsize: int = 1024) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._queues.append(q)
        return q

    def publish(self, event: VizEvent) -> None:
        for fn in list(self._handlers):
            try:
                fn(event)
            except Exception:  # pragma: no cover — handlers must not crash training
                pass
        for q in list(self._queues):
            try:
                q.put_nowait(event)
            except queue.Full:
                # drop oldest to keep live consumers responsive
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except queue.Empty:
                    pass
