from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, action_size):
        self.action_size = action_size
        self.training_rewards = []
        self.last_loss = None

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
