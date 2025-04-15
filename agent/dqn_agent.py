import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .base_agent import BaseAgent

class DQN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

class DeepQAgent(BaseAgent):
    def __init__(self, state_dim=3, gamma=0.99, eps_start=1.0, eps_min=0.01, eps_decay=0.995):
        super().__init__()
        self.policy_net = DQN(state_dim)
        self.target_net = DQN(state_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if np.random.random() < self.eps:
            return random.randint(0, 1)
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            return self.policy_net(state_tensor).argmax().item()

    def learn(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self._replay()

    def _replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eps = max(self.eps_min, self.eps * self.eps_decay)
