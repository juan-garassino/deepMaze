"""Shared plumbing for episode-buffer value agents (DRQN / DTQN).

Everything that is identical between the two memory agents lives here:
device/epsilon/gamma fields, the episode-buffer lifecycle (including the
end-of-episode flush that gets truncated episodes into replay), stored-vs-
grid observation tensors, the masked TD loss, and the optimize/clip/
target-sync tail. The subclasses keep only what genuinely differs: network
construction, move(), and the forward-specific middle of _learn().

The epsilon bug history is why this class exists — the per-step decay had
to be fixed in four places because this plumbing was copy-pasted.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from base_agent import BaseAgent
from episode_buffer import EpisodeBuffer


class RecurrentQAgent(BaseAgent):
    def __init__(self, action_size, grid_shape, *,
                 learning_rate, discount_factor, exploration_rate,
                 exploration_decay, min_epsilon, batch_size, seq_len,
                 burn_in, target_sync, buffer_capacity, learn_every,
                 aux_dim):
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
        self._learning_rate = learning_rate
        self._step = 0
        self.buf = EpisodeBuffer(buffer_capacity)

    # -- to be provided by subclasses -----------------------------------
    # self.model / self.target_model construction happens in the subclass
    # __init__, which then calls _finalize_models().

    def _finalize_models(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self._learning_rate)

    def _on_episode_boundary(self) -> None:
        """Reset per-episode inference state (hidden / context)."""
        raise NotImplementedError

    def _step_extra(self):
        """Optional per-transition payload stored in the buffer (DRQN's
        recurrent state for stored-state burn-in)."""
        return None

    def _learn(self) -> None:
        raise NotImplementedError

    # -- shared lifecycle ------------------------------------------------
    def on_episode_start(self):
        self._on_episode_boundary()
        self.buf.start_episode()

    def on_episode_end(self):
        # Flush truncated episodes (and the run's final episode) into the
        # buffer, then let the base class decay epsilon.
        self.buf.end_episode()
        super().on_episode_end()

    def update(self, state, action, reward, next_state, done, truncated=False):
        self.buf.add_step(state, action, reward, next_state, done,
                          extra=self._step_extra())
        self._step += 1
        if done:
            self._on_episode_boundary()
        if (len(self.buf) >= max(2, self.batch_size)
                and self._step % self.learn_every == 0):
            self._learn()

    # -- shared tensor / loss helpers -------------------------------------
    def _seq_tensor(self, arr):
        """Batch of stored observations → model input tensor (float when
        the obs carries aux features, long grid otherwise)."""
        if self.aux_dim:
            return torch.from_numpy(arr.astype(np.float32)).to(self.device)
        return torch.from_numpy(arr).long().to(self.device)

    def _obs_tensor(self, obs):
        """Single observation → (1, 1, ...) model input tensor."""
        if self.aux_dim:
            x = np.asarray(obs, dtype=np.float32).reshape(1, 1, -1)
            return torch.from_numpy(x).to(self.device)
        x = np.asarray(obs).reshape(1, 1, self.h, self.w)
        return torch.from_numpy(x).long().to(self.device)

    def _masked_td_loss(self, current_q, target_q, mask):
        """Masked MSE: padded positions (repeats of the terminal
        transition) carry no gradient."""
        td = current_q - target_q
        return (mask * td * td).sum() / mask.sum().clamp(min=1.0)

    def _optimize(self, loss) -> None:
        """Backward + clip + step + target sync; records last_loss."""
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.item())
        if self._step % self.target_sync == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def policy_snapshot(self):
        return {k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()}
