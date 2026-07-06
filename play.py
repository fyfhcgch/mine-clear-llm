import argparse
import json
import os

import torch
import numpy as np
from mine_env import MinesweeperEnv
from agents import DQNAgent, PPOAgent, GRPOAgent


def get_valid_mask(env):
    return env.valid_action_mask()


def load_run_config(checkpoint_path):
    config_path = os.path.join(os.path.dirname(checkpoint_path), "run_config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def check_checkpoint_config(parser, args):
    if not os.path.exists(args.checkpoint):
        parser.error(
            f"checkpoint not found: {args.checkpoint!r}. "
            "If you trained with SAVE_DIR=..., pass the same SAVE_DIR here, "
            "or pass CHECKPOINT=/path/to/best.pt or latest.pt."
        )

    if args.allow_config_mismatch:
        return

    config = load_run_config(args.checkpoint)
    if not config:
        return

    requested = {
        "algo": args.algo,
        "rows": args.rows,
        "cols": args.cols,
        "mines": args.mines,
        "action_mode": args.action_mode,
    }
    keys = ["algo", "rows", "cols", "mines", "action_mode"]
    mismatches = [k for k in keys if config.get(k) != requested.get(k)]
    if mismatches:
        details = ", ".join(
            f"{k}: checkpoint={config.get(k)!r}, requested={requested.get(k)!r}"
            for k in mismatches
        )
        suggestion = (
            f"ROWS={config.get('rows')} COLS={config.get('cols')} "
            f"MINES={config.get('mines')} ACTION_MODE={config.get('action_mode', 'reveal')}"
        )
        parser.error(
            "checkpoint config does not match play config: "
            f"{details}. Use matching make variables, e.g. {suggestion}, "
            "or pass --allow_config_mismatch if you intentionally want transfer/generalization."
        )


def make_agent(args, env):
    if args.algo == "dqn":
        return DQNAgent(n_actions=env.action_space.n, device=args.device)
    if args.algo == "ppo":
        return PPOAgent(n_actions=env.action_space.n, device=args.device)
    if args.algo == "grpo":
        return GRPOAgent(n_actions=env.action_space.n, device=args.device)
    raise ValueError(f"unsupported algo: {args.algo}")


def play(args):
    env = MinesweeperEnv(args.rows, args.cols, args.mines, render_mode="human", action_mode=args.action_mode)

    agent = make_agent(args, env)

    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    print(f"Loaded {args.algo.upper()} from {args.checkpoint}")
    print(f"Playing on {args.rows}x{args.cols} with {args.mines} mines\n")

    wins = 0
    games = args.games

    for game in range(1, games + 1):
        obs, _ = env.reset()
        env.render()
        done = False
        steps = 0

        while not done:
            valid_mask = get_valid_mask(env)

            if args.algo == "dqn":
                action = agent.select_action(obs, valid_mask)
            else:
                action, _, _ = agent.select_action(obs, valid_mask, deterministic=True)

            if env.action_mode == "reveal_flag" and action >= env.n_cells:
                r, c = divmod(action - env.n_cells, env.cols)
                print(f"Step {steps + 1}: Toggling flag ({r}, {c})")
            else:
                r, c = divmod(action, env.cols)
                print(f"Step {steps + 1}: Revealing ({r}, {c})")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            env.render()

            if done:
                if info.get("game_won"):
                    print("*** WIN! ***\n")
                    wins += 1
                else:
                    print("*** GAME OVER ***\n")
                break

    print(f"\nResults: {wins}/{games} wins ({wins/games*100:.1f}%)")


def evaluate(args):
    env = MinesweeperEnv(args.rows, args.cols, args.mines, action_mode=args.action_mode)

    agent = make_agent(args, env)

    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    print(f"Evaluating {args.algo.upper()} from {args.checkpoint}")
    print(f"Board: {args.rows}x{args.cols}, Mines: {args.mines}")
    print(f"Games: {args.games}\n")

    wins = 0
    total_reward = 0
    total_steps = 0
    rewards = []
    steps_list = []

    for game in range(1, args.games + 1):
        obs, _ = env.reset()
        done = False
        game_reward = 0
        steps = 0

        while not done:
            valid_mask = get_valid_mask(env)

            if args.algo == "dqn":
                action = agent.select_action(obs, valid_mask)
            else:
                action, _, _ = agent.select_action(obs, valid_mask, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            game_reward += reward
            steps += 1

        rewards.append(game_reward)
        steps_list.append(steps)
        total_reward += game_reward
        total_steps += steps

        if info.get("game_won"):
            wins += 1

        if game % 100 == 0:
            print(f"Game {game:5d} | "
                  f"Win Rate: {wins/game*100:5.1f}% | "
                  f"Avg Reward: {total_reward/game:.2f} | "
                  f"Avg Steps: {total_steps/game:.1f}")

    print(f"\nFinal Results:")
    print(f"Win Rate: {wins}/{args.games} ({wins/args.games*100:.1f}%)")
    print(f"Avg Reward: {total_reward/args.games:.2f}")
    print(f"Avg Steps: {total_steps/args.games:.1f}")
    print(f"Reward Std: {np.std(rewards):.2f}")


def validate_args(parser, args):
    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive integers")
    max_mines = args.rows * args.cols - 1
    if args.mines < 0 or args.mines > max_mines:
        parser.error(
            f"--mines must be between 0 and {max_mines} for a "
            f"{args.rows}x{args.cols} board; got {args.mines}"
        )
    if args.games <= 0:
        parser.error("--games must be a positive integer")
    check_checkpoint_config(parser, args)



def main():
    parser = argparse.ArgumentParser(description="Play Minesweeper with Trained Agent")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo", "grpo"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--action_mode", type=str, default="reveal", choices=["reveal", "reveal_flag"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--allow_config_mismatch", action="store_true",
                        help="Allow playing/evaluating a checkpoint on a different board config")

    args = parser.parse_args()
    validate_args(parser, args)

    if args.eval:
        evaluate(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
