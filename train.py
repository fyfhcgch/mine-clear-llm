import argparse
import json
import os
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agents import DQNAgent, GRPOAgent, PPOAgent
from mine_env import MinesweeperEnv


WINDOW = 100


def get_valid_mask(env):
    return env.valid_action_mask()


def get_save_dir(args, algo):
    return args.save_dir or os.path.join("checkpoints", algo)


def get_run_dir(args, algo):
    if args.run_name:
        return os.path.join("runs", args.run_name)
    return f"runs/{algo}_{args.rows}x{args.cols}_{args.mines}m"


def training_config(args, algo):
    return {
        "algo": algo,
        "rows": args.rows,
        "cols": args.cols,
        "mines": args.mines,
        "action_mode": args.action_mode,
        "n_cells": args.rows * args.cols,
        "total_safe_cells": args.rows * args.cols - args.mines,
    }


def save_training_state(save_dir, episode, best_reward, total_steps=0, best_win_rate=0.0, args=None, algo=None):
    state = {
        "episode": episode,
        "best_reward": float(best_reward) if np.isfinite(best_reward) else None,
        "best_win_rate": float(best_win_rate),
        "total_steps": total_steps,
    }
    if args is not None and algo is not None:
        state["config"] = training_config(args, algo)
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def save_run_config(save_dir, args, algo):
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(training_config(args, algo), f, indent=2)


def check_resume_config(parser, state, args, algo):
    old_config = state.get("config")
    if not old_config:
        return
    new_config = training_config(args, algo)
    keys = ["algo", "rows", "cols", "mines", "action_mode"]
    mismatches = [k for k in keys if old_config.get(k) != new_config.get(k)]
    if mismatches:
        details = ", ".join(
            f"{k}: checkpoint={old_config.get(k)!r}, requested={new_config.get(k)!r}"
            for k in mismatches
        )
        parser.error(
            "--resume checkpoint config does not match requested training config: "
            f"{details}. Use a new SAVE_DIR or pass matching ROWS/COLS/MINES."
        )


def load_training_state(save_dir):
    path = os.path.join(save_dir, "training_state.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def episode_progress(env):
    total_safe = env.rows * env.cols - env.n_mines
    safe_revealed = int(np.sum(env.revealed))
    return safe_revealed, safe_revealed / total_safe


def best_window_stats(rewards_history, win_history, allow_partial=False):
    if not rewards_history:
        return None
    if len(rewards_history) < rewards_history.maxlen and not allow_partial:
        return None
    return float(np.mean(rewards_history)), float(np.mean(win_history) * 100)


def format_best_reward(best_reward):
    return f"{best_reward:.2f}" if np.isfinite(best_reward) else "N/A"


def is_better_checkpoint(avg_reward, win_rate, best_reward, best_win_rate):
    # Prefer actual win rate over shaped reward.  Shaped reward can rise even if
    # the agent only opens several safe cells and then steps on a mine.
    return win_rate > best_win_rate or (
        np.isclose(win_rate, best_win_rate) and avg_reward > best_reward
    )


def update_best_checkpoint(agent, save_dir, rewards_history, win_history, best_reward, best_win_rate):
    stats = best_window_stats(rewards_history, win_history)
    if stats is None:
        return best_reward, best_win_rate

    avg_reward, win_rate = stats
    if is_better_checkpoint(avg_reward, win_rate, best_reward, best_win_rate):
        best_reward = avg_reward
        best_win_rate = win_rate
        agent.save(os.path.join(save_dir, "best.pt"))
    return best_reward, best_win_rate


def log_episode(writer, episode, total_reward, steps, won, safe_revealed, progress, loss=None, epsilon=None):
    writer.add_scalar("Reward/episode", total_reward, episode)
    writer.add_scalar("Steps/episode", steps, episode)
    writer.add_scalar("Win/episode", int(won), episode)
    writer.add_scalar("SafeCells/episode", safe_revealed, episode)
    writer.add_scalar("Progress/episode", progress, episode)
    if loss is not None:
        writer.add_scalar("Loss/latest", loss, episode)
    if epsilon is not None:
        writer.add_scalar("Epsilon", epsilon, episode)


def log_window(writer, episode, rewards_history, win_history, safe_history, progress_history, losses, epsilon=None):
    avg_reward = float(np.mean(rewards_history)) if rewards_history else 0.0
    win_rate = float(np.mean(win_history) * 100) if win_history else 0.0
    avg_safe = float(np.mean(safe_history)) if safe_history else 0.0
    avg_progress = float(np.mean(progress_history) * 100) if progress_history else 0.0
    avg_loss = float(np.mean(losses[-WINDOW:])) if losses else 0.0

    fields = [
        f"Episode {episode:5d}",
        f"Avg Reward: {avg_reward:7.2f}",
        f"Win Rate: {win_rate:5.1f}%",
        f"Progress: {avg_progress:5.1f}%",
        f"Safe: {avg_safe:5.1f}",
    ]
    if epsilon is not None:
        fields.append(f"Epsilon: {epsilon:.4f}")
    fields.append(f"Loss: {avg_loss:.4f}")
    print(" | ".join(fields))

    writer.add_scalar("Reward/avg_100", avg_reward, episode)
    writer.add_scalar("WinRate/avg_100", win_rate, episode)
    writer.add_scalar("Progress/avg_100", avg_progress, episode)
    writer.add_scalar("SafeCells/avg_100", avg_safe, episode)


def init_training(args, algo, agent):
    writer = SummaryWriter(get_run_dir(args, algo))
    save_dir = get_save_dir(args, algo)
    os.makedirs(save_dir, exist_ok=True)
    save_run_config(save_dir, args, algo)

    best_reward = -float("inf")
    best_win_rate = 0.0
    total_steps = 0
    start_episode = 1

    if args.resume:
        state = load_training_state(save_dir)
        if state:
            check_resume_config(args.parser, state, args, algo)
            start_episode = state["episode"] + 1
            loaded_best_reward = state.get("best_reward")
            if loaded_best_reward is not None:
                best_reward = loaded_best_reward
            best_win_rate = state.get("best_win_rate", 0.0)
            total_steps = state.get("total_steps", 0)
            agent.load(os.path.join(save_dir, "latest.pt"))
            print(
                f"Resuming from episode {state['episode']}, "
                f"best_reward: {best_reward:.2f}, best_win_rate: {best_win_rate:.1f}%"
            )
        else:
            print("No training state found, starting fresh")

    if start_episode > args.episodes:
        args.episodes = start_episode + 999
        print(f"Increased episodes to {args.episodes} for continued training")

    return writer, save_dir, best_reward, best_win_rate, total_steps, start_episode


def new_histories():
    return {
        "rewards": deque(maxlen=WINDOW),
        "wins": deque(maxlen=WINDOW),
        "safe": deque(maxlen=WINDOW),
        "progress": deque(maxlen=WINDOW),
    }


def record_episode(histories, total_reward, won, safe_revealed, progress):
    histories["rewards"].append(total_reward)
    histories["wins"].append(1 if won else 0)
    histories["safe"].append(safe_revealed)
    histories["progress"].append(progress)


def print_training_header(algo, args, agent, env, start_episode):
    total_safe = env.rows * env.cols - env.n_mines
    print(f"Training {algo.upper()} on {env.rows}x{env.cols} with {env.n_mines} mines")
    print(f"Action mode: {env.action_mode}; actions: {env.action_space.n}; total safe cells: {total_safe}")
    print(f"Device: {agent.device}")
    print(f"Episodes: {start_episode} -> {args.episodes}\n")


def train_dqn(args):
    env = MinesweeperEnv(args.rows, args.cols, args.mines, action_mode=args.action_mode)
    agent = DQNAgent(
        n_actions=env.action_space.n,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        device=args.device,
    )

    writer, save_dir, best_reward, best_win_rate, total_steps, start_episode = init_training(args, "dqn", agent)
    histories = new_histories()

    print_training_header("dqn", args, agent, env, start_episode)

    for episode in range(start_episode, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        info = {}

        while not done:
            valid_mask = get_valid_mask(env)
            action = agent.select_action(obs, valid_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_valid_mask = get_valid_mask(env) if not done else np.zeros(env.action_space.n, dtype=bool)
            agent.buffer.push(obs, action, reward, next_obs, float(done), next_valid_mask)
            agent.update()

            obs = next_obs
            total_reward += reward
            steps += 1
            total_steps += 1

        won = bool(info.get("game_won"))
        safe_revealed, progress = episode_progress(env)
        record_episode(histories, total_reward, won, safe_revealed, progress)
        best_reward, best_win_rate = update_best_checkpoint(
            agent, save_dir, histories["rewards"], histories["wins"], best_reward, best_win_rate
        )

        log_episode(
            writer, episode, total_reward, steps, won, safe_revealed, progress,
            loss=agent.losses[-1] if agent.losses else None,
            epsilon=agent.epsilon,
        )

        if episode % WINDOW == 0:
            log_window(
                writer, episode, histories["rewards"], histories["wins"], histories["safe"],
                histories["progress"], agent.losses, epsilon=agent.epsilon
            )

        if episode % args.save_interval == 0:
            agent.save(os.path.join(save_dir, f"ep_{episode}.pt"))
            agent.save(os.path.join(save_dir, "latest.pt"))
            save_training_state(save_dir, episode, best_reward, total_steps, best_win_rate, args, args.algo)

    agent.save(os.path.join(save_dir, "final.pt"))
    agent.save(os.path.join(save_dir, "latest.pt"))
    save_training_state(save_dir, episode, best_reward, total_steps, best_win_rate, args, args.algo)
    writer.close()
    print(f"\nTraining complete. Best win rate: {best_win_rate:.1f}%, best avg reward: {format_best_reward(best_reward)}")


def train_ppo(args):
    env = MinesweeperEnv(args.rows, args.cols, args.mines, action_mode=args.action_mode)
    agent = PPOAgent(
        n_actions=env.action_space.n,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef,
        device=args.device,
    )

    writer, save_dir, best_reward, best_win_rate, total_steps, start_episode = init_training(args, "ppo", agent)
    histories = new_histories()
    rollout_steps = 0

    print_training_header("ppo", args, agent, env, start_episode)

    for episode in range(start_episode, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        info = {}

        while not done:
            valid_mask = get_valid_mask(env)
            action, log_prob, value = agent.select_action(obs, valid_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.add(obs, action, reward, float(done), log_prob, value, valid_mask)

            obs = next_obs
            total_reward += reward
            steps += 1
            total_steps += 1
            rollout_steps += 1

        if rollout_steps >= args.update_steps:
            agent.update()
            rollout_steps = 0

        won = bool(info.get("game_won"))
        safe_revealed, progress = episode_progress(env)
        record_episode(histories, total_reward, won, safe_revealed, progress)
        best_reward, best_win_rate = update_best_checkpoint(
            agent, save_dir, histories["rewards"], histories["wins"], best_reward, best_win_rate
        )

        log_episode(
            writer, episode, total_reward, steps, won, safe_revealed, progress,
            loss=agent.losses[-1] if agent.losses else None,
        )

        if episode % WINDOW == 0:
            log_window(
                writer, episode, histories["rewards"], histories["wins"], histories["safe"],
                histories["progress"], agent.losses
            )

        if episode % args.save_interval == 0:
            agent.save(os.path.join(save_dir, f"ep_{episode}.pt"))
            agent.save(os.path.join(save_dir, "latest.pt"))
            save_training_state(save_dir, episode, best_reward, total_steps, best_win_rate, args, args.algo)

    if len(agent.buffer.observations) > 0:
        agent.update()
    agent.save(os.path.join(save_dir, "final.pt"))
    agent.save(os.path.join(save_dir, "latest.pt"))
    save_training_state(save_dir, episode, best_reward, total_steps, best_win_rate, args, args.algo)
    writer.close()
    print(f"\nTraining complete. Best win rate: {best_win_rate:.1f}%, best avg reward: {format_best_reward(best_reward)}")


def train_grpo(args):
    env = MinesweeperEnv(args.rows, args.cols, args.mines, action_mode=args.action_mode)
    agent = GRPOAgent(
        n_actions=env.action_space.n,
        lr=args.lr,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        group_size=args.group_size,
        epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        vf_coef=args.grpo_vf_coef if args.grpo_vf_coef is not None else args.vf_coef,
        device=args.device,
    )

    writer, save_dir, best_reward, best_win_rate, total_steps, start_episode = init_training(args, "grpo", agent)
    histories = new_histories()

    print_training_header("grpo", args, agent, env, start_episode)

    for episode in range(start_episode, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        info = {}
        episode_obs, episode_actions, episode_rewards = [], [], []
        episode_log_probs, episode_values, episode_masks = [], [], []

        while not done:
            valid_mask = get_valid_mask(env)
            action, log_prob, value = agent.select_action(obs, valid_mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_obs.append(obs)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            episode_values.append(value)
            episode_masks.append(valid_mask)

            obs = next_obs
            total_reward += reward
            steps += 1
            total_steps += 1

        agent.store_episode({
            "observations": episode_obs,
            "actions": episode_actions,
            "rewards": episode_rewards,
            "log_probs": episode_log_probs,
            "values": episode_values,
            "valid_masks": episode_masks,
        })

        if len(agent.episode_buffer) >= args.group_size:
            agent.update()

        won = bool(info.get("game_won"))
        safe_revealed, progress = episode_progress(env)
        record_episode(histories, total_reward, won, safe_revealed, progress)
        best_reward, best_win_rate = update_best_checkpoint(
            agent, save_dir, histories["rewards"], histories["wins"], best_reward, best_win_rate
        )

        log_episode(
            writer, episode, total_reward, steps, won, safe_revealed, progress,
            loss=agent.losses[-1] if agent.losses else None,
        )

        if episode % WINDOW == 0:
            log_window(
                writer, episode, histories["rewards"], histories["wins"], histories["safe"],
                histories["progress"], agent.losses
            )

        if episode % args.save_interval == 0:
            agent.save(os.path.join(save_dir, f"ep_{episode}.pt"))
            agent.save(os.path.join(save_dir, "latest.pt"))
            save_training_state(save_dir, episode, best_reward, total_steps, best_win_rate, args, args.algo)

    agent.save(os.path.join(save_dir, "final.pt"))
    agent.save(os.path.join(save_dir, "latest.pt"))
    save_training_state(save_dir, episode, best_reward, total_steps, best_win_rate, args, args.algo)
    writer.close()
    print(f"\nTraining complete. Best win rate: {best_win_rate:.1f}%, best avg reward: {format_best_reward(best_reward)}")


def validate_args(parser, args):
    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive integers")
    max_mines = args.rows * args.cols - 1
    if args.mines < 0 or args.mines > max_mines:
        parser.error(
            f"--mines must be between 0 and {max_mines} for a "
            f"{args.rows}x{args.cols} board; got {args.mines}"
        )
    if args.device and args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested with --device, but torch.cuda.is_available() is False")


def main():
    parser = argparse.ArgumentParser(description="Train Minesweeper RL Agent")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo", "grpo"])
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--action_mode", type=str, default="reveal", choices=["reveal", "reveal_flag"],
                        help="reveal keeps backward-compatible cell reveal actions; reveal_flag adds flag-toggle actions")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--epsilon_decay", type=float, default=0.9995)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--grpo_vf_coef", type=float, default=0.05,
                        help="Value loss coefficient for GRPO; lower than PPO because GRPO uses normalized return targets")
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--update_steps", type=int, default=2048)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional TensorBoard run subdirectory under runs/")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Optional checkpoint directory; defaults to checkpoints/<algo>")
    parser.add_argument("--device", type=str, default=None,
                        help="Training device, e.g. cuda, cuda:0, or cpu. Defaults to CUDA when available")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")

    args = parser.parse_args()
    args.parser = parser
    validate_args(parser, args)

    torch.manual_seed(42)
    np.random.seed(42)

    if args.algo == "dqn":
        train_dqn(args)
    elif args.algo == "ppo":
        train_ppo(args)
    elif args.algo == "grpo":
        train_grpo(args)


if __name__ == "__main__":
    main()
