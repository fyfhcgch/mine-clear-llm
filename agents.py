import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import DQNetwork, PPOActorCritic, ReplayBuffer, RolloutBuffer


ARCHITECTURE_VERSION = "spatial_v1"


def _load_state_dict_compatible(module, state_dict, checkpoint_path):
    try:
        module.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path!r} is not compatible with the current spatial_v1 model. "
            "Start a new SAVE_DIR/RUN_NAME instead of resuming old pooled-model checkpoints, "
            "or check out the older code that created the checkpoint."
        ) from exc


def _torch_load_checkpoint(path, device):
    # PyTorch 2.6+ defaults torch.load(weights_only=True), which can reject
    # optimizer checkpoints containing numpy scalar metadata.  These files are
    # produced locally by this trainer, so loading the full checkpoint is the
    # expected behavior.  Keep a fallback for older PyTorch versions that do not
    # support the weights_only keyword.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class DQNAgent:
    def __init__(self, n_actions=81, lr=1e-4, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9995, buffer_size=100000,
                 batch_size=64, target_update=100, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.learn_step = 0

        self.policy_net = DQNetwork(5, 256, n_actions).to(self.device)
        self.target_net = DQNetwork(5, 256, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.losses = []

    def select_action(self, obs, valid_mask=None):
        if np.random.random() < self.epsilon:
            if valid_mask is not None:
                valid_actions = np.where(valid_mask)[0]
                if len(valid_actions) > 0:
                    return int(np.random.choice(valid_actions))
            return int(np.random.randint(self.n_actions))

        with torch.no_grad():
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state).squeeze(0)

            if valid_mask is not None:
                mask = torch.BoolTensor(valid_mask).to(self.device)
                q_values = q_values.masked_fill(~mask, -torch.finfo(q_values.dtype).max)

            return int(q_values.argmax().item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, next_valid_masks = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        if next_valid_masks is not None:
            next_valid_masks = next_valid_masks.to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online network chooses the next action, target network evaluates it.
            next_policy_q = self.policy_net(next_states)
            if next_valid_masks is not None:
                next_policy_q = next_policy_q.masked_fill(
                    ~next_valid_masks, -torch.finfo(next_policy_q.dtype).max
                )
            next_actions = next_policy_q.argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        self.losses.append(loss.item())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            "architecture_version": ARCHITECTURE_VERSION,
            "n_actions": self.n_actions,
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step": self.learn_step,
        }, path)

    def load(self, path):
        checkpoint = _torch_load_checkpoint(path, self.device)
        _load_state_dict_compatible(self.policy_net, checkpoint["policy_net"], path)
        _load_state_dict_compatible(self.target_net, checkpoint["target_net"], path)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.learn_step = checkpoint.get("learn_step", self.learn_step)


class PPOAgent:
    def __init__(self, n_actions=81, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, epochs=4, batch_size=64,
                 entropy_coef=0.01, vf_coef=0.5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self.network = PPOActorCritic(5, 256, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.buffer = RolloutBuffer()
        self.losses = []

    def select_action(self, obs, valid_mask=None, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits, value = self.network(state)

            if valid_mask is not None:
                mask = torch.BoolTensor(valid_mask).to(self.device)
                logits = logits.masked_fill(~mask.unsqueeze(0), -torch.finfo(logits.dtype).max)

            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            return int(action.item()), float(dist.log_prob(action).item()), float(value.squeeze(-1).item())

    def update(self):
        if len(self.buffer.observations) == 0:
            return

        observations = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        valid_masks = self._buffer_valid_masks()

        returns = self.buffer.compute_returns(self.gamma).to(self.device)

        advantages = self._compute_gae()
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            indices = np.arange(len(self.buffer.observations))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                mask_batch = None if valid_masks is None else valid_masks[batch_idx]

                new_log_probs, entropy, values = self.network.evaluate(
                    observations[batch_idx], actions[batch_idx], mask_batch
                )

                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])

                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, returns[batch_idx])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                n_updates += 1

        self.buffer.clear()
        self.losses.append(total_loss / max(n_updates, 1))

    def _buffer_valid_masks(self):
        if not self.buffer.valid_masks or self.buffer.valid_masks[0] is None:
            return None
        return torch.BoolTensor(np.array(self.buffer.valid_masks)).to(self.device)

    def _compute_gae(self):
        advantages = []
        gae = 0.0
        values = self.buffer.values + [0.0]

        for t in reversed(range(len(self.buffer.rewards))):
            delta = (self.buffer.rewards[t] +
                     self.gamma * values[t + 1] * (1 - self.buffer.dones[t]) -
                     values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - self.buffer.dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def save(self, path):
        torch.save({
            "architecture_version": ARCHITECTURE_VERSION,
            "n_actions": self.n_actions,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = _torch_load_checkpoint(path, self.device)
        _load_state_dict_compatible(self.network, checkpoint["network"], path)
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class GRPOAgent:
    def __init__(self, n_actions=81, lr=3e-4, gamma=0.99, clip_epsilon=0.2,
                 group_size=8, epochs=4, batch_size=64,
                 entropy_coef=0.01, vf_coef=0.5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.group_size = group_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef

        self.network = PPOActorCritic(5, 256, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.episode_buffer = []
        self.losses = []

    def select_action(self, obs, valid_mask=None, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits, value = self.network(state)

            if valid_mask is not None:
                mask = torch.BoolTensor(valid_mask).to(self.device)
                logits = logits.masked_fill(~mask.unsqueeze(0), -torch.finfo(logits.dtype).max)

            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            return int(action.item()), float(dist.log_prob(action).item()), float(value.squeeze(-1).item())

    def store_episode(self, episode_data):
        self.episode_buffer.append(episode_data)

    def update(self):
        if len(self.episode_buffer) < self.group_size:
            return

        indices = np.random.choice(len(self.episode_buffer), self.group_size, replace=False)
        group_episode_returns = np.array(
            [sum(self.episode_buffer[i]["rewards"]) for i in indices], dtype=np.float32
        )
        group_mean = float(group_episode_returns.mean())

        all_obs, all_actions, all_old_log_probs, all_advantages, all_returns, all_masks = [], [], [], [], [], []

        for idx in indices:
            episode = self.episode_buffer[idx]
            rewards = episode["rewards"]
            n_steps = len(rewards)
            returns_to_go = self._discounted_returns(rewards)

            # Use per-step reward-to-go for credit assignment.  The previous
            # implementation assigned the same full episode return to every
            # action, which reinforced the final mine-click in otherwise good
            # episodes and made wins very hard to learn.
            step_advantages = returns_to_go - group_mean

            all_obs.append(np.array(episode["observations"]))
            all_actions.append(np.array(episode["actions"]))
            all_old_log_probs.append(np.array(episode.get("log_probs", np.zeros(n_steps)), dtype=np.float32))
            all_advantages.append(step_advantages.astype(np.float32))
            all_returns.append(returns_to_go.astype(np.float32))
            if "valid_masks" in episode:
                all_masks.append(np.array(episode["valid_masks"], dtype=bool))

        observations = torch.FloatTensor(np.concatenate(all_obs)).to(self.device)
        actions = torch.LongTensor(np.concatenate(all_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.concatenate(all_old_log_probs)).to(self.device)
        advantages = torch.FloatTensor(np.concatenate(all_advantages)).to(self.device)
        returns = torch.FloatTensor(np.concatenate(all_returns)).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        value_targets = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        valid_masks = None
        if all_masks:
            valid_masks = torch.BoolTensor(np.concatenate(all_masks)).to(self.device)

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            batch_indices = np.arange(len(observations))
            np.random.shuffle(batch_indices)

            for start in range(0, len(batch_indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = batch_indices[start:end]
                mask_batch = None if valid_masks is None else valid_masks[batch_idx]

                new_log_probs, entropy, values = self.network.evaluate(
                    observations[batch_idx], actions[batch_idx], mask_batch
                )
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, value_targets[batch_idx])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                n_updates += 1

        self.episode_buffer.clear()
        self.losses.append(total_loss / max(n_updates, 1))

    def _discounted_returns(self, rewards):
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = float(rewards[t]) + self.gamma * running
            returns[t] = running
        return returns

    def save(self, path):
        torch.save({
            "architecture_version": ARCHITECTURE_VERSION,
            "n_actions": self.n_actions,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = _torch_load_checkpoint(path, self.device)
        _load_state_dict_compatible(self.network, checkpoint["network"], path)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
