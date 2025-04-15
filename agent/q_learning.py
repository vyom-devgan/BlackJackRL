from collections import defaultdict
import numpy as np
from .base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__()
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice([0, 1])
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state]) if not done else 0
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state][action] = new_q

    def get_policy_table(self):
        """Convert tuple keys to strings for JSON serialization"""
        policy = {}
        for state, q_values in self.q_table.items():
            state_str = f"{state[0]},{state[1]},{state[2]}"  # Convert tuple to string
            policy[state_str] = {
                'hit_q': float(q_values[0]),  # Convert numpy float to native float
                'stand_q': float(q_values[1]),
                'best_action': "HIT" if q_values[0] > q_values[1] else "STAND"
            }
        return policy
