"""Episode-granular replay buffer shared by DRQN and DTQN.

Capacity is measured in EPISODES (default 200 ≈ 120k transitions at
600-step episodes), not transitions.

Per-step record: (obs, prev_action, action, reward, next_obs, done).
prev_action is the action taken INTO this state; NO_ACTION (-1) marks the
episode start.

sample() pads episodes shorter than seq_len by repeating the final
transition (tail-only padding) and returns a validity mask so consumers
exclude the pads from the loss — without the mask, short successful
episodes train mostly on duplicated terminal transitions.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

NO_ACTION = -1


class EpisodeBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: deque = deque(maxlen=capacity)
        self._current: list[tuple] = []
        self._prev_a: int = NO_ACTION

    def start_episode(self) -> None:
        """Flush any leftover steps and reset prev-action tracking."""
        self.end_episode()

    def end_episode(self) -> None:
        """Commit the in-progress episode (truncation / final episode of a
        run). Idempotent — a no-op when the episode already flushed on done."""
        if self._current:
            self.episodes.append(self._current)
        self._current = []
        self._prev_a = NO_ACTION

    def add_step(self, obs, action, reward, next_obs, done) -> None:
        self._current.append((
            np.asarray(obs), int(self._prev_a), int(action),
            float(reward), np.asarray(next_obs), float(done),
        ))
        self._prev_a = int(action)
        if done:
            self.end_episode()

    def __len__(self) -> int:
        """Number of COMPLETED episodes."""
        return len(self.episodes)

    def sample(self, batch_size: int, seq_len: int) -> dict[str, np.ndarray]:
        eps = random.sample(self.episodes, min(batch_size, len(self.episodes)))
        seqs, masks = [], []
        for ep in eps:
            if len(ep) <= seq_len:
                pad = [ep[-1]] * (seq_len - len(ep))
                seqs.append(ep + pad)
                masks.append([1.0] * len(ep) + [0.0] * (seq_len - len(ep)))
            else:
                start = random.randint(0, len(ep) - seq_len)
                seqs.append(ep[start:start + seq_len])
                masks.append([1.0] * seq_len)
        return {
            "obs": np.stack([[s[0] for s in seq] for seq in seqs]),
            "prev_action": np.array([[s[1] for s in seq] for seq in seqs],
                                    dtype=np.int64),
            "action": np.array([[s[2] for s in seq] for seq in seqs],
                               dtype=np.int64),
            "reward": np.array([[s[3] for s in seq] for seq in seqs],
                               dtype=np.float32),
            "next_obs": np.stack([[s[4] for s in seq] for seq in seqs]),
            "done": np.array([[s[5] for s in seq] for seq in seqs],
                             dtype=np.float32),
            "mask": np.array(masks, dtype=np.float32),
        }
