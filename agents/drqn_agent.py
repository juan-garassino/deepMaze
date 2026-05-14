"""DRQN — DQN with a recurrent (LSTM) memory.

Memory matters when the observation is partial (`--partial K`): the agent
only sees a small window around itself, and the LSTM lets it remember
where it has already been. Online inference keeps a hidden state across
`move()` calls within an episode; the bus / train loop calls
`on_episode_start()` to reset.

Training: a per-episode buffer; sample `batch_size` episodes, take a
random `seq_len`-step sub-sequence per episode, burn in the first half,
learn on the second half.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from base_agent import BaseAgent
from nets import VOCAB, encode_grid_batch


class DRQN(nn.Module):
    def __init__(self, h: int, w: int, action_size: int,
                 lstm_hidden: int = 128):
        super().__init__()
        self.h, self.w = h, w
        self.lstm_hidden = lstm_hidden
        self.conv = nn.Sequential(
            nn.Conv2d(VOCAB, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden,
                            batch_first=True)
        self.head = nn.Linear(lstm_hidden, action_size)

    def forward(self, x_seq: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        """x_seq: (B, T, h, w) ints or (B, T, VOCAB, h, w) floats.
        Returns (q_seq, (h_n, c_n))."""
        if x_seq.dim() == 4:
            # (B, T, h, w) -> one-hot
            B, T, H, W = x_seq.shape
            flat = x_seq.reshape(B * T, H, W).cpu().numpy()
            x = encode_grid_batch(flat, H, W).to(x_seq.device)
        else:
            B, T = x_seq.shape[:2]
            x = x_seq.reshape(B * T, *x_seq.shape[2:])
        z = self.conv(x).reshape(B, T, -1)
        out, hidden = self.lstm(z, hidden)
        return self.head(out), hidden


class EpisodeBuffer:
    """Stores complete episodes as parallel arrays."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: deque = deque(maxlen=capacity)
        self._current: list[tuple] = []

    def add_step(self, obs, action, reward, next_obs, done):
        self._current.append((np.asarray(obs), int(action), float(reward),
                              np.asarray(next_obs), float(done)))
        if done:
            self.commit()

    def commit(self):
        if self._current:
            self.episodes.append(self._current)
            self._current = []

    def __len__(self):
        return len(self.episodes)

    def sample(self, batch_size: int, seq_len: int):
        eps = random.sample(self.episodes, min(batch_size, len(self.episodes)))
        seqs = []
        for ep in eps:
            if len(ep) <= seq_len:
                # pad at end by repeating last transition w/ done=1
                pad = [ep[-1]] * (seq_len - len(ep))
                seqs.append(ep + pad)
            else:
                start = random.randint(0, len(ep) - seq_len)
                seqs.append(ep[start:start + seq_len])
        # transpose to per-step arrays
        obs = np.stack([[s[0] for s in seq] for seq in seqs])
        act = np.array([[s[1] for s in seq] for seq in seqs], dtype=np.int64)
        rew = np.array([[s[2] for s in seq] for seq in seqs], dtype=np.float32)
        nobs = np.stack([[s[3] for s in seq] for seq in seqs])
        done = np.array([[s[4] for s in seq] for seq in seqs], dtype=np.float32)
        return obs, act, rew, nobs, done


class DRQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, grid_shape, learning_rate=1e-3,
                 discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.995, min_epsilon=0.05,
                 batch_size=8, seq_len=8, burn_in=4, target_sync=100,
                 buffer_capacity=200, lstm_hidden=128):
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

        self.model = DRQN(self.h, self.w, action_size, lstm_hidden).to(self.device)
        self.target_model = DRQN(self.h, self.w, action_size,
                                 lstm_hidden).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buf = EpisodeBuffer(buffer_capacity)
        self._hidden = None

    # ------------------------------------------------------------------
    def on_episode_start(self):
        self._hidden = None

    def _obs_tensor(self, obs):
        x = np.asarray(obs).reshape(1, 1, self.h, self.w)
        return torch.from_numpy(x).long().to(self.device)

    def move(self, state):
        x = self._obs_tensor(state)
        with torch.no_grad():
            q, self._hidden = self.model(x.float(), self._hidden)
        if (not self.deterministic) and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        return int(q.squeeze(0).squeeze(0).argmax().item())

    def update(self, state, action, reward, next_state, done):
        self.buf.add_step(state, action, reward, next_state, done)
        self.add_training_reward(reward)
        self._step += 1
        if done:
            self._hidden = None
        if len(self.buf) >= max(2, self.batch_size):
            self._learn()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _learn(self):
        obs, act, rew, nobs, dn = self.buf.sample(self.batch_size, self.seq_len)
        B, T, H, W = obs.shape
        obs_t = torch.from_numpy(obs).long().to(self.device).float()
        nobs_t = torch.from_numpy(nobs).long().to(self.device).float()
        act_t = torch.from_numpy(act).to(self.device)
        rew_t = torch.from_numpy(rew).to(self.device)
        dn_t = torch.from_numpy(dn).to(self.device)

        # Burn-in first half to set hidden state without learning.
        bi = min(self.burn_in, T - 1)
        with torch.no_grad():
            _, hid = self.model(obs_t[:, :bi], None)
            _, hid_tgt = self.target_model(nobs_t[:, :bi], None)

        q_seq, _ = self.model(obs_t[:, bi:], hid)
        with torch.no_grad():
            qt_seq, _ = self.target_model(nobs_t[:, bi:], hid_tgt)
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
    def q_values(self, state):
        # Note: ignores hidden — use only for viz / policy heatmap.
        x = self._obs_tensor(state).float()
        with torch.no_grad():
            q, _ = self.model(x, None)
        return q.squeeze(0).squeeze(0).cpu().numpy()

    def q_values_batch(self, states):
        x = np.stack([np.asarray(s).reshape(1, self.h, self.w) for s in states])
        x = torch.from_numpy(x).long().to(self.device).float()
        with torch.no_grad():
            q, _ = self.model(x, None)
        return q.squeeze(1).cpu().numpy()

    def policy_snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
