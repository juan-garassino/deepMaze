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


def encode_grid_batch(states: np.ndarray, h: int, w: int) -> torch.Tensor:
    """`states` shape (N, h*w) or (N, h, w) of integer cell labels.
    Returns float tensor (N, VOCAB, h, w) one-hot encoded."""
    x = np.asarray(states)
    if x.ndim == 2 and x.shape[1] == h * w:
        x = x.reshape(-1, h, w)
    n = x.shape[0]
    flat = np.clip(x, 0, VOCAB - 1).astype(np.int64).reshape(n, -1)
    onehot = np.zeros((n, flat.shape[1], VOCAB), dtype=np.float32)
    onehot[np.arange(n)[:, None], np.arange(flat.shape[1])[None, :], flat] = 1.0
    return torch.from_numpy(onehot.reshape(n, h, w, VOCAB).transpose(0, 3, 1, 2))


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
            x = encode_grid_batch(x.cpu().numpy(), self.h, self.w).to(x.device)
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
            x = encode_grid_batch(x.cpu().numpy(), self.h, self.w).to(x.device)
        z = self.trunk(x)
        return torch.softmax(self.actor(z), dim=-1), self.critic(z)


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
        return torch.softmax(self.actor(h), dim=-1), self.critic(h)
