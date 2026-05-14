from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, action_size):
        self.action_size = action_size
        self.training_rewards = []
        self.last_loss = None
        self.deterministic = False
        self._saved_epsilon = None

    def set_deterministic(self, flag: bool) -> None:
        """Toggle greedy action selection for evaluation/replay."""
        flag = bool(flag)
        if flag and not self.deterministic and hasattr(self, "epsilon"):
            self._saved_epsilon = self.epsilon
            self.epsilon = 0.0
        elif (not flag) and self.deterministic and self._saved_epsilon is not None \
                and hasattr(self, "epsilon"):
            self.epsilon = self._saved_epsilon
            self._saved_epsilon = None
        self.deterministic = flag

    def q_values_batch(self, states):
        """Vectorised q_values. Default: Python loop. Net agents override."""
        import numpy as np
        return np.stack([self.q_values(s) for s in states])

    @abstractmethod
    def move(self, state):
        ...

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        ...

    def q_values(self, state):
        return None

    def policy_snapshot(self):
        return None

    def add_training_reward(self, reward):
        self.training_rewards.append(reward)
