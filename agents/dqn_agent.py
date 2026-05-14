import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from base_agent import BaseAgent
from replay_buffer import ReplayBuffer


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size, learning_rate=1e-3, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, min_epsilon=0.01,
                 batch_size=64, target_sync=200, buffer_capacity=10000):
        super().__init__(action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_sync = target_sync
        self._step = 0

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_capacity, state_size, 1)

    def _to_tensor(self, state):
        return torch.from_numpy(np.asarray(state, dtype=np.float32).flatten()).unsqueeze(0).to(self.device)

    def move(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        with torch.no_grad():
            q = self.model(self._to_tensor(state))
        return int(q.argmax(dim=1).item())

    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self._step += 1

        if len(self.memory) >= self.batch_size:
            batch = self.memory.sample(self.batch_size)
            s = torch.from_numpy(batch['state']).to(self.device)
            a = torch.from_numpy(batch['action']).to(self.device)
            r = torch.from_numpy(batch['reward']).to(self.device)
            ns = torch.from_numpy(batch['next_state']).to(self.device)
            d = torch.from_numpy(batch['done']).to(self.device)

            current_q = self.model(s).gather(1, a)
            with torch.no_grad():
                next_q = self.target_model(ns).max(1, keepdim=True)[0]
                target = r + (1.0 - d) * self.gamma * next_q

            loss = nn.functional.mse_loss(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.item())

            if self._step % self.target_sync == 0:
                self.target_model.load_state_dict(self.model.state_dict())

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.add_training_reward(reward)

    def q_values(self, state):
        with torch.no_grad():
            return self.model(self._to_tensor(state)).cpu().numpy().flatten()

    def q_values_batch(self, states):
        flat = np.stack([np.asarray(s, dtype=np.float32).flatten() for s in states])
        x = torch.from_numpy(flat).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy()

    def policy_snapshot(self):
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
