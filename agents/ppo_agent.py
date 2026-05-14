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
    def __init__(self, state_size, action_size, learning_rate=3e-4, discount_factor=0.99,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 update_frequency=128, epochs=4, gae_lambda=0.95):
        super().__init__(action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.gamma = discount_factor
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_frequency = update_frequency
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        self.epsilon = 0.0  # for uniform interface with eps-greedy logging

        self.ac = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=learning_rate)
        self._buf = []

    def _flat(self, state):
        return np.asarray(state, dtype=np.float32).flatten()

    def move(self, state):
        s = torch.from_numpy(self._flat(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.ac(s)
        dist = Categorical(probs)
        a = dist.sample()
        self._last_logprob = dist.log_prob(a).item()
        return int(a.item())

    def store_transition(self, state, action, reward, next_state, done):
        self._buf.append((self._flat(state), int(action), float(reward),
                          self._flat(next_state), float(done), self._last_logprob))

    def update(self, state, action, reward, next_state, done):
        self.store_transition(state, action, reward, next_state, done)
        self.add_training_reward(reward)
        if len(self._buf) >= self.update_frequency or done:
            self._learn()

    def _learn(self):
        if not self._buf:
            return
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

        advantages = torch.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            delta = r[t] + self.gamma * next_values[t] * (1 - d[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - d[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        losses = []
        for _ in range(self.epochs):
            probs, vals = self.ac(s)
            vals = vals.squeeze(-1)
            dist = Categorical(probs)
            new_lp = dist.log_prob(a)
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(vals, returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.optimizer.step()
            losses.append(float(loss.item()))

        self.last_loss = float(np.mean(losses)) if losses else None
        self._buf.clear()

    def q_values(self, state):
        s = torch.from_numpy(self._flat(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs, _ = self.ac(s)
        return probs.cpu().numpy().flatten()

    def policy_snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.ac.state_dict().items()}
