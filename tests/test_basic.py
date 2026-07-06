import unittest

import numpy as np
import torch

from agents import GRPOAgent, PPOAgent
from mine_env import MinesweeperEnv
from models import DQNetwork, PPOActorCritic


class BasicRegressionTests(unittest.TestCase):
    def test_env_seed_reproducible(self):
        env = MinesweeperEnv(9, 9, 10)
        env.reset(seed=123)
        board1 = env.board.copy()
        env.reset(seed=123)
        board2 = env.board.copy()
        self.assertTrue(np.array_equal(board1, board2))

    def test_first_reveal_is_safe(self):
        for seed in range(20):
            env = MinesweeperEnv(9, 9, 10)
            env.reset(seed=seed)
            _, _, terminated, _, info = env.step(0)
            self.assertFalse(info.get("game_over", False))
            self.assertFalse(terminated and info.get("game_over", False))

    def test_reveal_and_reveal_flag_action_spaces(self):
        reveal_env = MinesweeperEnv(4, 4, 2)
        flag_env = MinesweeperEnv(4, 4, 2, action_mode="reveal_flag")
        self.assertEqual(reveal_env.action_space.n, 16)
        self.assertEqual(flag_env.action_space.n, 32)
        self.assertEqual(reveal_env.valid_action_mask().shape, (16,))
        self.assertEqual(flag_env.valid_action_mask().shape, (32,))

    def test_models_support_multiple_board_sizes(self):
        for rows, cols, actions in [(4, 4, 16), (9, 9, 81), (16, 16, 256)]:
            dqn = DQNetwork(n_actions=actions)
            ac = PPOActorCritic(n_actions=actions)
            x = torch.zeros(2, 5, rows, cols)
            self.assertEqual(dqn(x).shape, (2, actions))
            logits, values = ac(x)
            self.assertEqual(logits.shape, (2, actions))
            self.assertEqual(values.shape, (2, 1))

    def test_ppo_update_smoke(self):
        agent = PPOAgent(n_actions=16, batch_size=2, epochs=1)
        obs = np.zeros((5, 4, 4), dtype=np.float32)
        valid_mask = np.ones(16, dtype=bool)
        agent.buffer.add(obs, 0, 1.0, 0.0, 0.0, 0.0, valid_mask)
        agent.buffer.add(obs, 1, -1.0, 1.0, 0.0, 0.0, valid_mask)
        agent.update()
        self.assertEqual(len(agent.buffer.observations), 0)
        self.assertEqual(len(agent.losses), 1)


    def test_grpo_update_smoke(self):
        agent = GRPOAgent(n_actions=16, group_size=2, batch_size=2, epochs=1)
        obs = np.zeros((5, 4, 4), dtype=np.float32)
        valid_mask = np.ones(16, dtype=bool)
        agent.store_episode({
            "observations": [obs, obs],
            "actions": [0, 1],
            "rewards": [1.0, -10.0],
            "log_probs": [0.0, 0.0],
            "valid_masks": [valid_mask, valid_mask],
        })
        agent.store_episode({
            "observations": [obs, obs],
            "actions": [2, 3],
            "rewards": [1.0, 100.0],
            "log_probs": [0.0, 0.0],
            "valid_masks": [valid_mask, valid_mask],
        })
        agent.update()
        self.assertEqual(len(agent.episode_buffer), 0)
        self.assertEqual(len(agent.losses), 1)


    def test_ppo_gae_runs_backward(self):
        agent = PPOAgent(n_actions=16, gamma=1.0, gae_lambda=1.0)
        agent.buffer.rewards = [1.0, 1.0, 1.0]
        agent.buffer.dones = [0.0, 0.0, 1.0]
        agent.buffer.values = [0.0, 0.0, 0.0]
        self.assertEqual(agent._compute_gae(), [3.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
