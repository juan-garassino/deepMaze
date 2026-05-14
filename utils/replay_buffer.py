import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity, state_size, action_dim=1):
        self.capacity = capacity
        self.state_size = state_size
        self.action_dim = action_dim
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32).flatten(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).flatten(),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return {
            'state': np.stack(s),
            'action': np.array(a, dtype=np.int64).reshape(-1, 1),
            'reward': np.array(r, dtype=np.float32).reshape(-1, 1),
            'next_state': np.stack(ns),
            'done': np.array(d, dtype=np.float32).reshape(-1, 1),
        }

    def __len__(self):
        return len(self.buffer)
