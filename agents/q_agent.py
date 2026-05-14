from collections import defaultdict

import numpy as np
from base_agent import BaseAgent


class QAgent(BaseAgent):
    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_epsilon=0.01):
        super().__init__(action_size)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_epsilon
        self.Q = defaultdict(lambda: np.zeros(self.action_size))

    def _key(self, state):
        return tuple(np.asarray(state).flatten().tolist())

    def move(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        return int(np.argmax(self.Q[self._key(state)]))

    def update(self, state, action, reward, next_state, done):
        k, nk = self._key(state), self._key(next_state)
        current_q = self.Q[k][action]
        next_q = 0.0 if done else float(np.max(self.Q[nk]))
        td = reward + self.gamma * next_q - current_q
        self.Q[k][action] = current_q + self.lr * td
        self.last_loss = float(td * td)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.add_training_reward(reward)

    def q_values(self, state):
        return self.Q[self._key(state)].copy()

    def policy_snapshot(self):
        return {k: v.copy() for k, v in self.Q.items()}
