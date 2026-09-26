"""
DDPG actor-critic agent (CM3020 Draft Report, Chapter 3.2; algorithm from
Lillicrap et al., 2015 -- Chapter 2.1.4), sized for CPU training on a laptop.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)  # portfolio weights on the simplex (Chapter 3.2)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """
    Critic loss: L(theta^Q) = E[(Q(s_t,a_t|theta^Q) - y_t)^2]   (Chapter 3.2)
    Actor update: deterministic policy gradient, maximise Q(s, actor(s))
    """

    def __init__(self, state_dim, action_dim, gamma=0.99, tau=0.005, lr=1e-4, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def act(self, state, noise_scale=0.0):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            weights = self.actor(state_t).cpu().numpy()[0]
        if noise_scale > 0:
            weights = weights + np.random.normal(0, noise_scale, size=weights.shape)
            weights = np.clip(weights, 0, None)
            weights = weights / (weights.sum() + 1e-8)
        return weights

    def _soft_update(self, source, target):
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(s2)
            target_q = self.critic_target(s2, next_action)
            y = r + self.gamma * (1 - d) * target_q
        q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return float(critic_loss.item()), float(actor_loss.item())

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
