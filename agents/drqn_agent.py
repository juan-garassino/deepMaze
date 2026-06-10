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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from base_agent import BaseAgent
from encoders import GridAttnEncoder
from episode_buffer import NO_ACTION, EpisodeBuffer


class DRQN(nn.Module):
    def __init__(self, h: int, w: int, action_size: int,
                 enc_dim: int = 64, lstm_hidden: int = 128,
                 action_emb_dim: int = 16, aux_dim: int = 0):
        super().__init__()
        self.h, self.w = h, w
        self.action_size = action_size
        self.aux_dim = aux_dim
        self.enc = GridAttnEncoder(h, w, dim=enc_dim, aux_dim=aux_dim)
        # +1 for NO_ACTION sentinel
        self.action_emb = nn.Embedding(action_size + 1, action_emb_dim)
        self.lstm = nn.LSTM(enc_dim + action_emb_dim, lstm_hidden,
                            batch_first=True)
        self.head = nn.Linear(lstm_hidden, action_size)

    def _encode(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq: (B, T, h, w) int grid OR (B, T, h*w+aux) flat float.
        Returns (B, T, enc_dim)."""
        if obs_seq.dim() == 3:  # flat grid+aux form
            B, T, F = obs_seq.shape
            gl = self.h * self.w
            grid = obs_seq[..., :gl].reshape(B * T, self.h, self.w).long()
            aux = obs_seq[..., gl:].reshape(B * T, -1).float()
            return self.enc(grid, aux).reshape(B, T, -1)
        B, T, H, W = obs_seq.shape
        return self.enc(obs_seq.reshape(B * T, H, W)).reshape(B, T, -1)

    def forward(self, obs_seq: torch.Tensor, prev_actions: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        """obs_seq: (B, T, h, w) long or (B, T, h*w+aux) float.
        prev_actions: (B, T) long (-1 = none)."""
        feats = self._encode(obs_seq)                           # (B, T, enc_dim)
        a_idx = (prev_actions + 1).clamp(min=0)                 # shift -1 -> 0
        a_emb = self.action_emb(a_idx)                          # (B, T, ae)
        z = torch.cat([feats, a_emb], dim=-1)
        out, hidden = self.lstm(z, hidden)
        return self.head(out), hidden


class DRQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, grid_shape, learning_rate=1e-3,
                 discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995, min_epsilon=0.05,
                 batch_size=8, seq_len=8, burn_in=4, target_sync=100,
                 buffer_capacity=200, lstm_hidden=128, enc_dim=64,
                 action_emb_dim=16, aux_dim=0, learn_every=1):
        super().__init__(action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h, self.w = grid_shape
        self.aux_dim = aux_dim
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.target_sync = target_sync
        self.learn_every = max(1, int(learn_every))
        self._step = 0

        self.model = DRQN(self.h, self.w, action_size, enc_dim, lstm_hidden,
                          action_emb_dim, aux_dim).to(self.device)
        self.target_model = DRQN(self.h, self.w, action_size, enc_dim,
                                 lstm_hidden, action_emb_dim,
                                 aux_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buf = EpisodeBuffer(buffer_capacity)
        self._hidden = None
        self._last_action = NO_ACTION

    # ------------------------------------------------------------------
    def on_episode_start(self):
        self._hidden = None
        self._last_action = NO_ACTION
        self.buf.start_episode()

    def on_episode_end(self):
        # Flush truncated episodes (and the run's final episode) into the
        # buffer, then let the base class decay epsilon.
        self.buf.end_episode()
        super().on_episode_end()

    def _obs_tensor(self, obs):
        if self.aux_dim:
            x = np.asarray(obs, dtype=np.float32).reshape(1, 1, -1)
            return torch.from_numpy(x).to(self.device)
        x = np.asarray(obs).reshape(1, 1, self.h, self.w)
        return torch.from_numpy(x).long().to(self.device)

    def _seq_tensor(self, arr):
        """Batch of stored observations → model input tensor."""
        if self.aux_dim:
            return torch.from_numpy(arr.astype(np.float32)).to(self.device)
        return torch.from_numpy(arr).long().to(self.device)

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
        if (len(self.buf) >= max(2, self.batch_size)
                and self._step % self.learn_every == 0):
            self._learn()

    def _learn(self):
        batch = self.buf.sample(self.batch_size, self.seq_len)
        obs_t = self._seq_tensor(batch["obs"])
        nobs_t = self._seq_tensor(batch["next_obs"])
        pa_t = torch.from_numpy(batch["prev_action"]).to(self.device)
        act_t = torch.from_numpy(batch["action"]).to(self.device)
        rew_t = torch.from_numpy(batch["reward"]).to(self.device)
        dn_t = torch.from_numpy(batch["done"]).to(self.device)
        mask_t = torch.from_numpy(batch["mask"]).to(self.device)
        T = obs_t.shape[1]
        # next-step "prev action" is the current action at each step
        npa_t = act_t

        bi = min(self.burn_in, T - 1)
        with torch.no_grad():
            _, hid = self.model(obs_t[:, :bi], pa_t[:, :bi], None)
            _, hid_nxt = self.model(nobs_t[:, :bi], npa_t[:, :bi], None)
            _, hid_tgt = self.target_model(nobs_t[:, :bi], npa_t[:, :bi], None)

        q_seq, _ = self.model(obs_t[:, bi:], pa_t[:, bi:], hid)
        with torch.no_grad():
            # Double DQN: online net (with its own next-obs hidden) selects,
            # target net evaluates.
            qn_on, _ = self.model(nobs_t[:, bi:], npa_t[:, bi:], hid_nxt)
            next_a = qn_on.argmax(dim=-1, keepdim=True)
            qt_seq, _ = self.target_model(nobs_t[:, bi:], npa_t[:, bi:], hid_tgt)
            max_next = qt_seq.gather(2, next_a).squeeze(-1)

        a_use = act_t[:, bi:]
        r_use = rew_t[:, bi:]
        d_use = dn_t[:, bi:]
        m_use = mask_t[:, bi:]
        current_q = q_seq.gather(2, a_use.unsqueeze(-1)).squeeze(-1)
        target_q = r_use + (1 - d_use) * self.gamma * max_next

        # Masked MSE: padded positions (repeats of the terminal transition)
        # carry no gradient.
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
        if self.aux_dim:
            x = np.stack([np.asarray(s, dtype=np.float32).reshape(1, -1)
                          for s in states])
            x = torch.from_numpy(x).to(self.device)
        else:
            x = np.stack([np.asarray(s).reshape(1, self.h, self.w)
                          for s in states])
            x = torch.from_numpy(x).long().to(self.device)
        pa = torch.full((x.shape[0], 1), NO_ACTION,
                        dtype=torch.long, device=self.device)
        with torch.no_grad():
            q, _ = self.model(x, pa, None)
        return q.squeeze(1).cpu().numpy()

    def policy_snapshot(self):
        return {k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()}
