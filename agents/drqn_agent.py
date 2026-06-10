"""DRQN — DQN with a recurrent (LSTM) memory.

v2 changes vs v1:
- Per-step encoder is now GridAttnEncoder (spatial self-attention over the
  partial view). The LSTM sees richer per-step features.
- Prev-action embedding concatenated to the LSTM input. Helps the LSTM
  encode "I just moved north", which is essential for backtracking out of
  dead ends under partial observation.

Online inference keeps a hidden state across `move()` calls within an
episode; `on_episode_start()` resets it. Training: per-episode buffer,
sample sub-sequences, burn-in then learn.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from base_agent import BaseAgent
from encoders import GridAttnEncoder

# Per-step buffer record: (obs, prev_action, action, reward, next_obs, done).
# prev_action is the action taken into this state; -1 means "no prev action".
NO_ACTION = -1


class DRQN(nn.Module):
    def __init__(self, h: int, w: int, action_size: int,
                 enc_dim: int = 64, lstm_hidden: int = 128,
                 action_emb_dim: int = 16):
        super().__init__()
        self.h, self.w = h, w
        self.action_size = action_size
        self.enc = GridAttnEncoder(h, w, dim=enc_dim)
        # +1 for NO_ACTION sentinel
        self.action_emb = nn.Embedding(action_size + 1, action_emb_dim)
        self.lstm = nn.LSTM(enc_dim + action_emb_dim, lstm_hidden,
                            batch_first=True)
        self.head = nn.Linear(lstm_hidden, action_size)

    def forward(self, obs_seq: torch.Tensor, prev_actions: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        """obs_seq: (B, T, h, w) long. prev_actions: (B, T) long (-1 = none)."""
        B, T, H, W = obs_seq.shape
        flat = obs_seq.reshape(B * T, H, W)
        feats = self.enc(flat).reshape(B, T, -1)               # (B, T, enc_dim)
        a_idx = (prev_actions + 1).clamp(min=0)                 # shift -1 -> 0
        a_emb = self.action_emb(a_idx)                          # (B, T, ae)
        z = torch.cat([feats, a_emb], dim=-1)
        out, hidden = self.lstm(z, hidden)
        return self.head(out), hidden


class EpisodeBuffer:
    """Stores complete episodes with prev_action tracking."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: deque = deque(maxlen=capacity)
        self._current: list[tuple] = []
        self._prev_a: int = NO_ACTION

    def reset_episode(self):
        if self._current:
            self.episodes.append(self._current)
        self._current = []
        self._prev_a = NO_ACTION

    def add_step(self, obs, action, reward, next_obs, done):
        self._current.append((
            np.asarray(obs), int(self._prev_a), int(action),
            float(reward), np.asarray(next_obs), float(done),
        ))
        self._prev_a = int(action)
        if done:
            self.episodes.append(self._current)
            self._current = []
            self._prev_a = NO_ACTION

    def __len__(self):
        return len(self.episodes)

    def sample(self, batch_size: int, seq_len: int):
        eps = random.sample(self.episodes, min(batch_size, len(self.episodes)))
        seqs = []
        for ep in eps:
            if len(ep) <= seq_len:
                pad = [ep[-1]] * (seq_len - len(ep))
                seqs.append(ep + pad)
            else:
                start = random.randint(0, len(ep) - seq_len)
                seqs.append(ep[start:start + seq_len])
        obs = np.stack([[s[0] for s in seq] for seq in seqs])
        pa = np.array([[s[1] for s in seq] for seq in seqs], dtype=np.int64)
        act = np.array([[s[2] for s in seq] for seq in seqs], dtype=np.int64)
        rew = np.array([[s[3] for s in seq] for seq in seqs], dtype=np.float32)
        nobs = np.stack([[s[4] for s in seq] for seq in seqs])
        done = np.array([[s[5] for s in seq] for seq in seqs], dtype=np.float32)
        return obs, pa, act, rew, nobs, done


class DRQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, grid_shape, learning_rate=1e-3,
                 discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995, min_epsilon=0.05,
                 batch_size=8, seq_len=8, burn_in=4, target_sync=100,
                 buffer_capacity=200, lstm_hidden=128, enc_dim=64,
                 action_emb_dim=16):
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
        self._step = 0

        self.model = DRQN(self.h, self.w, action_size, enc_dim, lstm_hidden,
                          action_emb_dim).to(self.device)
        self.target_model = DRQN(self.h, self.w, action_size, enc_dim,
                                 lstm_hidden, action_emb_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buf = EpisodeBuffer(buffer_capacity)
        self._hidden = None
        self._last_action = NO_ACTION

    # ------------------------------------------------------------------
    def on_episode_start(self):
        self._hidden = None
        self._last_action = NO_ACTION
        self.buf.reset_episode()

    def _obs_tensor(self, obs):
        x = np.asarray(obs).reshape(1, 1, self.h, self.w)
        return torch.from_numpy(x).long().to(self.device)

    def move(self, state):
        x = self._obs_tensor(state)
        pa = torch.tensor([[self._last_action]],
                          dtype=torch.long, device=self.device)
        with torch.no_grad():
            q, self._hidden = self.model(x, pa, self._hidden)
        if (not self.deterministic) and np.random.random() < self.epsilon:
            a = int(np.random.randint(0, self.action_size))
        else:
            a = int(q.squeeze(0).squeeze(0).argmax().item())
        self._last_action = a
        return a

    def update(self, state, action, reward, next_state, done, truncated=False):
        self.buf.add_step(state, action, reward, next_state, done)
        self._step += 1
        if done:
            self._hidden = None
            self._last_action = NO_ACTION
        if len(self.buf) >= max(2, self.batch_size):
            self._learn()

    def _learn(self):
        obs, pa, act, rew, nobs, dn = self.buf.sample(
            self.batch_size, self.seq_len)
        B, T, H, W = obs.shape
        obs_t = torch.from_numpy(obs).long().to(self.device)
        nobs_t = torch.from_numpy(nobs).long().to(self.device)
        pa_t = torch.from_numpy(pa).to(self.device)
        act_t = torch.from_numpy(act).to(self.device)
        rew_t = torch.from_numpy(rew).to(self.device)
        dn_t = torch.from_numpy(dn).to(self.device)
        # next-step "prev action" is the current action at each step
        npa_t = act_t

        bi = min(self.burn_in, T - 1)
        with torch.no_grad():
            _, hid = self.model(obs_t[:, :bi], pa_t[:, :bi], None)
            _, hid_tgt = self.target_model(nobs_t[:, :bi], npa_t[:, :bi], None)

        q_seq, _ = self.model(obs_t[:, bi:], pa_t[:, bi:], hid)
        with torch.no_grad():
            qt_seq, _ = self.target_model(nobs_t[:, bi:], npa_t[:, bi:], hid_tgt)
            max_next = qt_seq.max(dim=-1).values

        a_use = act_t[:, bi:]
        r_use = rew_t[:, bi:]
        d_use = dn_t[:, bi:]
        current_q = q_seq.gather(2, a_use.unsqueeze(-1)).squeeze(-1)
        target_q = r_use + (1 - d_use) * self.gamma * max_next

        loss = nn.functional.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.item())

        if self._step % self.target_sync == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    # ------------------------------------------------------------------
    def memory_snapshot(self):
        """Return a small float vector summarising current memory.
        For DRQN: the LSTM hidden state h (clipped to ~32 dim for the strip viz)."""
        if self._hidden is None:
            return None
        h = self._hidden[0]  # (num_layers, B=1, hidden)
        flat = h.detach().cpu().numpy().reshape(-1)
        return {"kind": "lstm_hidden", "data": flat[:64].tolist()}

    def q_values(self, state):
        x = self._obs_tensor(state)
        pa = torch.tensor([[NO_ACTION]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q, _ = self.model(x, pa, None)
        return q.squeeze(0).squeeze(0).cpu().numpy()

    def q_values_batch(self, states):
        x = np.stack([np.asarray(s).reshape(1, self.h, self.w) for s in states])
        x = torch.from_numpy(x).long().to(self.device)
        pa = torch.full((x.shape[0], 1), NO_ACTION,
                        dtype=torch.long, device=self.device)
        with torch.no_grad():
            q, _ = self.model(x, pa, None)
        return q.squeeze(1).cpu().numpy()

    def policy_snapshot(self):
        return {k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()}
