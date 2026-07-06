from collections import deque

import numpy as np
import torch
import torch.nn as nn


class SpatialFeatureExtractor(nn.Module):
    """Fully-convolutional encoder for Minesweeper observations.

    The previous pooled+MLP encoder compressed the whole board to a 2x2 feature
    map before producing 81 action logits.  That loses most cell-level spatial
    information, while Minesweeper actions are inherently per-cell decisions.

    This encoder preserves the HxW grid all the way to the policy/Q heads.  It
    still supports any board size because the action heads are convolutional and
    flatten their per-cell logits at runtime.
    """

    def __init__(self, in_channels=5, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# Backward-compatible alias used by older imports/tests.
CNNFeatureExtractor = SpatialFeatureExtractor


def _flatten_action_planes(action_map, n_actions):
    """Convert [B, planes, H, W] to [B, n_actions].

    Supported action spaces are reveal-only (H*W actions) and reveal+flag
    (2*H*W actions).  The head always emits two planes; reveal-only uses the
    first plane.
    """
    batch, planes_available, rows, cols = action_map.shape
    cells = rows * cols
    if n_actions == cells:
        planes = 1
    elif n_actions == 2 * cells:
        planes = 2
    else:
        raise ValueError(
            f"n_actions={n_actions} is incompatible with observation shape "
            f"{rows}x{cols}; expected {cells} or {2 * cells}"
        )
    if planes > planes_available:
        raise ValueError(f"action head has {planes_available} planes, need {planes}")
    return action_map[:, :planes].reshape(batch, planes * cells)


class DQNetwork(nn.Module):
    def __init__(self, in_channels=5, hidden=128, n_actions=81):
        super().__init__()
        self.n_actions = n_actions
        self.features = SpatialFeatureExtractor(in_channels, hidden)
        self.advantage = nn.Conv2d(hidden, 2, 1)
        self.value = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        features = self.features(x)
        adv = _flatten_action_planes(self.advantage(features), self.n_actions)
        val = self.value(features)
        return val + adv - adv.mean(dim=1, keepdim=True)


class PPOActorCritic(nn.Module):
    def __init__(self, in_channels=5, hidden=128, n_actions=81):
        super().__init__()
        self.n_actions = n_actions
        self.features = SpatialFeatureExtractor(in_channels, hidden)
        self.actor = nn.Conv2d(hidden, 2, 1)
        self.critic = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        features = self.features(x)
        logits = _flatten_action_planes(self.actor(features), self.n_actions)
        value = self.critic(features)
        return logits, value

    def get_action(self, obs, valid_mask=None):
        logits, value = self.forward(obs)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask.bool(), -torch.finfo(logits.dtype).max)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, obs, actions, valid_masks=None):
        logits, value = self.forward(obs)
        if valid_masks is not None:
            logits = logits.masked_fill(~valid_masks.bool(), -torch.finfo(logits.dtype).max)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_valid_mask=None):
        self.buffer.append((state, action, reward, next_state, done, next_valid_mask))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones, next_valid_masks = zip(
            *[self.buffer[i] for i in batch]
        )
        masks = None if next_valid_masks[0] is None else torch.BoolTensor(np.array(next_valid_masks))
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            masks,
        )

    def __len__(self):
        return len(self.buffer)


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.valid_masks = []

    def add(self, obs, action, reward, done, log_prob, value, valid_mask=None):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.valid_masks.append(None if valid_mask is None else np.array(valid_mask, dtype=bool))

    def compute_returns(self, gamma=0.99):
        returns = []
        R = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + gamma * R * (1 - done)
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def clear(self):
        self.__init__()
