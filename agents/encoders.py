"""Per-step observation encoders shared by DRQN and DTQN.

GridAttnEncoder: turns a (h, w) int maze view into a fixed-dim vector via
    one-hot encoding → conv stem → spatial multi-head self-attention with
    a CLS-style learnable query → pool.

Spatial attention here is cheap (25 tokens for a 5x5 partial view) but
real value — it lets the agent learn which cells in the window matter
for the current decision, before any temporal aggregator sees the result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from nets import VOCAB, grid_onehot


class GridAttnEncoder(nn.Module):
    def __init__(self, h: int, w: int, dim: int = 64, heads: int = 4,
                 conv_layers: int = 2):
        super().__init__()
        self.h, self.w = h, w
        self.dim = dim
        layers: list[nn.Module] = [nn.Conv2d(VOCAB, dim, 3, padding=1), nn.ReLU()]
        for _ in range(conv_layers - 1):
            layers += [nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        # learnable spatial position embedding
        self.pos = nn.Parameter(torch.zeros(1, h * w, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, h, w) int labels OR (B, VOCAB, h, w) one-hot float.
        Returns (B, dim)."""
        if x.dim() == 3:
            x = grid_onehot(x, self.h, self.w)
        B = x.shape[0]
        z = self.conv(x)                              # (B, dim, h, w)
        z = z.flatten(2).transpose(1, 2)              # (B, h*w, dim)
        z = z + self.pos
        cls = self.cls.expand(B, -1, -1)              # (B, 1, dim)
        out, _ = self.attn(cls, z, z, need_weights=False)
        return self.norm(out.squeeze(1))              # (B, dim)
