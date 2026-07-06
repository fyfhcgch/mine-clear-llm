import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mine_env import MinesweeperEnv
from expert import MinesweeperExpert, generate_demonstrations
from models import PPOActorCritic


class BehavioralCloningAgent:
    def __init__(self, n_actions=81, lr=1e-3, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.network = PPOActorCritic(5, 256, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def train_on_demonstrations(self, demonstrations, epochs=10, batch_size=64):
        all_obs = []
        all_actions = []

        for demo in demonstrations:
            for obs, action in zip(demo["observations"], demo["actions"]):
                all_obs.append(obs)
                all_actions.append(action)

        observations = torch.FloatTensor(np.array(all_obs)).to(self.device)
        actions = torch.LongTensor(all_actions).to(self.device)

        dataset = TensorDataset(observations, actions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for obs_batch, action_batch in dataloader:
                logits, _ = self.network(obs_batch)
                loss = nn.CrossEntropyLoss()(logits, action_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def select_action(self, obs, valid_mask=None):
        with torch.no_grad():
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits, _ = self.network(state)

            if valid_mask is not None:
                mask = torch.BoolTensor(valid_mask).to(self.device)
                logits[0][~mask] = -float("inf")

            return logits.argmax().item()

    def save(self, path):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def evaluate_agent(agent, env, n_episodes=100):
    wins = 0
    total_reward = 0
    total_steps = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            valid_mask = env.valid_action_mask()

            action = agent.select_action(obs, valid_mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        total_reward += episode_reward
        total_steps += steps
        if info.get("game_won"):
            wins += 1

    return {
        "win_rate": wins / n_episodes * 100,
        "avg_reward": total_reward / n_episodes,
        "avg_steps": total_steps / n_episodes,
    }


def validate_args(parser, args):
    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive integers")
    max_mines = args.rows * args.cols - 1
    if args.mines < 0 or args.mines > max_mines:
        parser.error(
            f"--mines must be between 0 and {max_mines} for a "
            f"{args.rows}x{args.cols} board; got {args.mines}"
        )
    if args.n_demos <= 0:
        parser.error("--n_demos must be a positive integer")
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")



def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning for Minesweeper")
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--n_demos", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_games", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc/best.pt")

    args = parser.parse_args()
    validate_args(parser, args)

    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs("checkpoints/bc", exist_ok=True)

    env = MinesweeperEnv(args.rows, args.cols, args.mines)
    expert = MinesweeperExpert(args.rows, args.cols)

    print(f"Generating {args.n_demos} expert demonstrations...")
    demonstrations = generate_demonstrations(env, expert, args.n_demos)

    agent = BehavioralCloningAgent(
        n_actions=env.action_space.n,
        lr=args.lr,
    )

    print(f"\nTraining behavioral cloning agent...")
    agent.train_on_demonstrations(demonstrations, epochs=args.epochs, batch_size=args.batch_size)

    print(f"\nEvaluating agent...")
    results = evaluate_agent(agent, env, args.eval_games)

    print(f"\nResults:")
    print(f"  Win Rate: {results['win_rate']:.1f}%")
    print(f"  Avg Reward: {results['avg_reward']:.2f}")
    print(f"  Avg Steps: {results['avg_steps']:.1f}")

    agent.save(args.checkpoint)
    print(f"\nModel saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
