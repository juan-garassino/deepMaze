"""Network factories for DQN / PPO / DRQN.

The CNN path expects the observation to be a 2-D maze grid (full view or
egocentric window). It converts integer cell labels into a stack of
one-hot channels — strong inductive bias for grid tasks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Cell-value vocabulary used to size the one-hot channel stack.
#   0 HOLE  1 LAND  2 START  3 EXIT  4 LAVA  5+ AGENT
VOCAB = 6


def grid_onehot(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`x`: (N, h*w) or (N, h, w) tensor of integer cell labels, any numeric
    dtype, any device. Returns (N, VOCAB, h, w) float32 one-hot ON THE SAME
    DEVICE — no CPU/numpy round trip per forward."""
    x = x.reshape(-1, h, w).long().clamp(0, VOCAB - 1)
    return torch.nn.functional.one_hot(x, VOCAB).permute(0, 3, 1, 2).float()


def encode_grid_batch(states: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Numpy wrapper around grid_onehot (kept for non-tensor callers)."""
    return grid_onehot(torch.from_numpy(np.asarray(states).astype(np.int64)),
                       h, w)


class MLPHead(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)


class CNNHead(nn.Module):
    """3 conv layers + global avg pool + linear. Output_size = action_size."""

    def __init__(self, h: int, w: int, output_size: int):
        super().__init__()
        self.h, self.w = h, w
        self.body = nn.Sequential(
            nn.Conv2d(VOCAB, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(32, output_size)

    def forward(self, x):
        # x: flattened ints of shape (N, h*w) or already a 4-D float tensor.
        if x.dim() == 2:
            x = grid_onehot(x, self.h, self.w)
        return self.head(self.body(x))


class CNNActorCritic(nn.Module):
    """Shared conv trunk + separate actor / critic heads."""

    def __init__(self, h: int, w: int, action_size: int):
        super().__init__()
        self.h, self.w = h, w
        self.trunk = nn.Sequential(
            nn.Conv2d(VOCAB, 16, kernel_size=3, padding=1), nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.Tanh(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.actor = nn.Linear(32, action_size)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = grid_onehot(x, self.h, self.w)
        z = self.trunk(x)
        return self.actor(z), self.critic(z)  # raw logits


class MLPActorCritic(nn.Module):
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)  # raw logits
