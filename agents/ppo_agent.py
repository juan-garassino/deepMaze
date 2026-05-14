import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from base_agent import BaseAgent


class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_size, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, output_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return torch.softmax(self.actor(h), dim=-1), self.critic(h)


class PPOAgent(BaseAgent):
    """Standard PPO with fixed-horizon rollouts + GAE + K-epoch M-minibatch updates.

    Rollouts span episode boundaries; GAE resets at `done`. Updates run when
    the buffer hits `n_steps`, not at episode end — that was the bug in the
    earlier version (only ~60 updates in 60 episodes).
    """

    def __init__(self, state_size, action_size, learning_rate=3e-4,
                 discount_factor=0.99, clip_eps=0.2, value_coef=0.5,
                 entropy_coef=0.01, n_steps=256, epochs=4, minibatches=4,
                 gae_lambda=0.95):
        super().__init__(action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.gamma = discount_factor
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_steps = int(n_steps)
        self.epochs = int(epochs)
        self.minibatches = int(minibatches)
        self.gae_lambda = gae_lambda
        self.epsilon = 0.0  # log-only; PPO is on-policy stochastic

        self.ac = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=learning_rate)
        self._buf = []
        self._last_logprob = 0.0

    # ------------------------------------------------------------------
    def _flat(self, state):
        return np.asarray(state, dtype=np.float32).flatten()

    def move(self, state):
        s = torch.from_numpy(self._flat(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.ac(s)
        if self.deterministic:
            a = int(probs.argmax(dim=-1).item())
            self._last_logprob = float(torch.log(probs[0, a] + 1e-12).item())
            return a
        dist = Categorical(probs)
        a = dist.sample()
        self._last_logprob = float(dist.log_prob(a).item())
        return int(a.item())

    def update(self, state, action, reward, next_state, done):
        self._buf.append((
            self._flat(state), int(action), float(reward),
            self._flat(next_state), float(done), self._last_logprob,
        ))
        self.add_training_reward(reward)
        if len(self._buf) >= self.n_steps:
            self._learn()

    def flush(self):
        if self._buf:
            self._learn()

    def _learn(self):
        s, a, r, ns, d, old_lp = zip(*self._buf)
        s = torch.from_numpy(np.stack(s)).to(self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        ns = torch.from_numpy(np.stack(ns)).to(self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)
        old_lp = torch.tensor(old_lp, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, values = self.ac(s)
            _, next_values = self.ac(ns)
            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)

        # GAE; resets at done.
        T = r.shape[0]
        advantages = torch.zeros(T, device=self.device)
        gae = torch.zeros((), device=self.device)
        for t in reversed(range(T)):
            nonterminal = 1.0 - d[t]
            delta = r[t] + self.gamma * next_values[t] * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        idx = np.arange(T)
        mb_size = max(1, T // self.minibatches)
        losses = []
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, T, mb_size):
                mb = idx[start:start + mb_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                probs, vals = self.ac(s[mb_t])
                vals = vals.squeeze(-1)
                dist = Categorical(probs)
                new_lp = dist.log_prob(a[mb_t])
                ratio = torch.exp(new_lp - old_lp[mb_t])
                surr1 = ratio * advantages[mb_t]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * advantages[mb_t]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(vals, returns[mb_t])
                entropy = dist.entropy().mean()
                loss = (actor_loss
                        + self.value_coef * critic_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()
                losses.append(float(loss.item()))

        self.last_loss = float(np.mean(losses)) if losses else None
        self._buf.clear()

    # ------------------------------------------------------------------
    def q_values(self, state):
        s = torch.from_numpy(self._flat(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.ac(s)
        return probs.cpu().numpy().flatten()

    def q_values_batch(self, states):
        flat = np.stack([self._flat(s) for s in states])
        x = torch.from_numpy(flat).to(self.device)
        with torch.no_grad():
            probs, _ = self.ac(x)
        return probs.cpu().numpy()

    def policy_snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.ac.state_dict().items()}
