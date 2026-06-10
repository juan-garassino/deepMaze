"""DTQN — Deep Transformer Q-Network.

Memory mechanism is a causal Transformer over the trajectory instead of an
LSTM. Per-step encoding is the same `GridAttnEncoder` (so the comparison
DRQN vs DTQN is an honest A/B on the temporal aggregator).

Per-token representation:
    token_t = encoder(obs_t) + action_emb(prev_action_t) + pos_emb(t)
Causal-masked transformer encoder layers → Q-head at each position.

Online inference: maintain a deque of last `max_ctx` (obs, prev_action)
pairs; re-encode each step. `on_episode_start()` clears the deque.

Training: episode buffer; sample `seq_len` sub-sequences; train Q-loss on
positions after a burn-in prefix.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from base_agent import BaseAgent
from encoders import GridAttnEncoder
from episode_buffer import NO_ACTION, EpisodeBuffer


class TransformerBlock(nn.Module):
    """Pre-norm transformer block exposing attention weights for viz."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.last_attn: torch.Tensor | None = None  # (B, T, T) per-block weights

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        out, w = self.attn(h, h, h, attn_mask=mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, average_attn_weights=True)
        self.last_attn = w.detach()
        x = x + out
        x = x + self.ff(self.ln2(x))
        return x


class DTQN(nn.Module):
    def __init__(self, h: int, w: int, action_size: int,
                 dim: int = 128, heads: int = 4, layers: int = 2,
                 max_ctx: int = 64, ff_mult: int = 2):
        super().__init__()
        self.h, self.w = h, w
        self.action_size = action_size
        self.dim = dim
        self.max_ctx = max_ctx

        self.enc = GridAttnEncoder(h, w, dim=dim)
        self.action_emb = nn.Embedding(action_size + 1, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_ctx, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, ff_mult) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, action_size)

    @staticmethod
    def causal_mask(T: int, device) -> torch.Tensor:
        # bool mask (True = disallowed) so it composes with the bool
        # key_padding_mask without dtype-mismatch warnings
        m = torch.ones((T, T), dtype=torch.bool, device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, obs_seq: torch.Tensor, prev_actions: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None):
        """obs_seq: (B, T, h, w) long. prev_actions: (B, T) long (-1 = none).
        key_padding_mask: optional (B, T) bool, True = padded position.
        Returns Q (B, T, A)."""
        B, T, H, W = obs_seq.shape
        if T > self.max_ctx:
            raise ValueError(f"context {T} > max_ctx {self.max_ctx}")
        flat = obs_seq.reshape(B * T, H, W)
        feats = self.enc(flat).reshape(B, T, -1)             # (B, T, dim)
        a_emb = self.action_emb((prev_actions + 1).clamp(min=0))
        pos = self.pos_emb[:, :T, :]
        z = feats + a_emb + pos
        mask = self.causal_mask(T, z.device)
        for blk in self.blocks:
            z = blk(z, mask, key_padding_mask)
        return self.head(self.ln_f(z))

    def last_layer_attention(self) -> torch.Tensor | None:
        """Attention weights from the last block, last forward pass."""
        return self.blocks[-1].last_attn


class DTQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, grid_shape,
                 learning_rate=3e-4, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995,
                 min_epsilon=0.05, batch_size=8, seq_len=16, burn_in=4,
                 target_sync=100, buffer_capacity=200,
                 dim=128, heads=4, layers=2, max_ctx=64):
        super().__init__(action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h, self.w = grid_shape
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.target_sync = target_sync
        self.max_ctx = max_ctx
        self._step = 0

        self.model = DTQN(self.h, self.w, action_size, dim, heads, layers,
                          max_ctx).to(self.device)
        self.target_model = DTQN(self.h, self.w, action_size, dim, heads,
                                 layers, max_ctx).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.buf = EpisodeBuffer(buffer_capacity)
        # inference context
        self._ctx_obs: deque = deque(maxlen=max_ctx)
        self._ctx_pa: deque = deque(maxlen=max_ctx)
        self._last_action = NO_ACTION

    # ------------------------------------------------------------------
    def on_episode_start(self):
        self._clear_context()
        self.buf.start_episode()

    def on_episode_end(self):
        # Flush truncated episodes (and the run's final episode) into the
        # buffer, then let the base class decay epsilon.
        self.buf.end_episode()
        super().on_episode_end()

    def _clear_context(self):
        self._ctx_obs.clear()
        self._ctx_pa.clear()
        self._last_action = NO_ACTION

    def _ctx_tensors(self):
        obs = np.stack(list(self._ctx_obs)).reshape(1, -1, self.h, self.w)
        pa = np.array([list(self._ctx_pa)], dtype=np.int64)
        return (torch.from_numpy(obs).long().to(self.device),
                torch.from_numpy(pa).to(self.device))

    def move(self, state):
        self._ctx_obs.append(np.asarray(state))
        self._ctx_pa.append(self._last_action)
        obs_t, pa_t = self._ctx_tensors()
        with torch.no_grad():
            q_seq = self.model(obs_t, pa_t)
        q_last = q_seq[0, -1]
        if (not self.deterministic) and np.random.random() < self.epsilon:
            a = int(np.random.randint(0, self.action_size))
        else:
            a = int(q_last.argmax().item())
        self._last_action = a
        return a

    def update(self, state, action, reward, next_state, done, truncated=False):
        self.buf.add_step(state, action, reward, next_state, done)
        self._step += 1
        if done:
            self._clear_context()  # episode already committed by add_step
        if len(self.buf) >= max(2, self.batch_size):
            self._learn()

    def _learn(self):
        batch = self.buf.sample(self.batch_size, self.seq_len)
        T = min(batch["obs"].shape[1], self.max_ctx)
        obs_t = torch.from_numpy(batch["obs"][:, :T]).long().to(self.device)
        nobs_t = torch.from_numpy(batch["next_obs"][:, :T]).long().to(self.device)
        pa_t = torch.from_numpy(batch["prev_action"][:, :T]).to(self.device)
        act_t = torch.from_numpy(batch["action"][:, :T]).to(self.device)
        rew_t = torch.from_numpy(batch["reward"][:, :T]).to(self.device)
        dn_t = torch.from_numpy(batch["done"][:, :T]).to(self.device)
        mask_t = torch.from_numpy(batch["mask"][:, :T]).to(self.device)
        npa_t = act_t  # next-step prev_action = current action
        # Pads are tail-only, so real positions never attend to garbage and
        # no row is fully masked (position 0 is always real).
        kpm = mask_t == 0  # (B, T) bool, True = padded

        q_seq = self.model(obs_t, pa_t, key_padding_mask=kpm)
        with torch.no_grad():
            # Double DQN: online net selects, target net evaluates.
            qn_on = self.model(nobs_t, npa_t, key_padding_mask=kpm)
            next_a = qn_on.argmax(dim=-1, keepdim=True)
            qt_seq = self.target_model(nobs_t, npa_t, key_padding_mask=kpm)
            max_next = qt_seq.gather(2, next_a).squeeze(-1)

        # Train only on positions after burn-in.
        bi = min(self.burn_in, T - 1)
        a_use = act_t[:, bi:]
        current_q = q_seq[:, bi:].gather(2, a_use.unsqueeze(-1)).squeeze(-1)
        target_q = rew_t[:, bi:] + (1 - dn_t[:, bi:]) * self.gamma * max_next[:, bi:]

        # Masked MSE: padded positions carry no gradient.
        m_use = mask_t[:, bi:]
        td = current_q - target_q
        loss = (m_use * td * td).sum() / m_use.sum().clamp(min=1.0)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.item())

        if self._step % self.target_sync == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    # ------------------------------------------------------------------
    def memory_snapshot(self):
        """Attention weights from the last forward pass: shape (T,) — what
        the current (last) token attends to over the context.
        Returns None if no forward has happened in this episode yet."""
        attn = self.model.last_layer_attention()
        if attn is None:
            return None
        last_row = attn[0, -1].cpu().numpy()  # (T,)
        return {"kind": "attention_row", "data": last_row.tolist()}

    def q_values(self, state):
        x = np.asarray(state).reshape(1, 1, self.h, self.w)
        x = torch.from_numpy(x).long().to(self.device)
        pa = torch.tensor([[NO_ACTION]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.model(x, pa)
        return q.squeeze(0).squeeze(0).cpu().numpy()

    def q_values_batch(self, states):
        x = np.stack([np.asarray(s).reshape(1, self.h, self.w) for s in states])
        x = torch.from_numpy(x).long().to(self.device)
        pa = torch.full((x.shape[0], 1), NO_ACTION,
                        dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = self.model(x, pa)
        return q.squeeze(1).cpu().numpy()

    def policy_snapshot(self):
        return {k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()}
