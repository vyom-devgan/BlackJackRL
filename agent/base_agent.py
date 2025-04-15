from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass

    def update_stats(self, reward):
        if reward == 1:
            self.wins += 1
        elif reward == -1:
            self.losses += 1
        else:
            self.draws += 1

    @property
    def win_rate(self):
        total = self.wins + self.losses + self.draws
        return self.wins / total if total else 0
