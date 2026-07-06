import numpy as np


class MinesweeperExpert:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def get_action(self, env):
        if env.done:
            return None

        safe, mines = self._analyze_constraints(env)

        if safe:
            r, c = safe[0]
            return r * self.cols + c

        if mines:
            r, c = mines[0]
            env.flagged[r, c] = True
            return self.get_action(env)

        boundary = self._get_boundary(env)
        if boundary:
            r, c = boundary[np.random.randint(len(boundary))]
            return r * self.cols + c

        for r in range(self.rows):
            for c in range(self.cols):
                if not env.revealed[r, c] and not env.flagged[r, c]:
                    return r * self.cols + c
        return None

    def _analyze_constraints(self, env):
        safe = set()
        mines = set()

        for r in range(self.rows):
            for c in range(self.cols):
                if not env.revealed[r, c] or env.board[r, c] <= 0:
                    continue

                unrevealed = []
                flagged = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if env.flagged[nr, nc]:
                                flagged += 1
                            elif not env.revealed[nr, nc]:
                                unrevealed.append((nr, nc))

                remaining = env.board[r, c] - flagged

                if remaining == 0 and unrevealed:
                    safe.update(unrevealed)
                elif remaining == len(unrevealed) and unrevealed:
                    mines.update(unrevealed)

        return list(safe), list(mines)

    def _get_boundary(self, env):
        boundary = []
        for r in range(self.rows):
            for c in range(self.cols):
                if env.revealed[r, c] and env.board[r, c] > 0:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < self.rows and 0 <= nc < self.cols
                                    and not env.revealed[nr, nc]
                                    and not env.flagged[nr, nc]):
                                if (nr, nc) not in boundary:
                                    boundary.append((nr, nc))
        return boundary


def generate_demonstrations(env, expert, n_episodes=1000):
    demonstrations = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        info = {}
        trajectory = {"observations": [], "actions": [], "flags": []}

        while not env.done:
            flag_actions = []
            safe, mines = expert._analyze_constraints(env)

            for r, c in mines:
                if not env.flagged[r, c]:
                    env.flagged[r, c] = True
                    flag_actions.append(r * env.cols + c)

            # Store the observation after deterministic expert flags are applied;
            # otherwise the supervised observation/action pair is inconsistent.
            obs_for_action = env._get_obs()
            safe, _ = expert._analyze_constraints(env)

            if safe:
                action = safe[0][0] * env.cols + safe[0][1]
            else:
                boundary = expert._get_boundary(env)
                if boundary:
                    r, c = boundary[np.random.randint(len(boundary))]
                    action = r * env.cols + c
                else:
                    found = False
                    for r in range(env.rows):
                        for c in range(env.cols):
                            if not env.revealed[r, c] and not env.flagged[r, c]:
                                action = r * env.cols + c
                                found = True
                                break
                        if found:
                            break
                    else:
                        break

            trajectory["observations"].append(obs_for_action)
            trajectory["actions"].append(action)
            trajectory["flags"].append(flag_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        trajectory["won"] = info.get("game_won", False)
        demonstrations.append(trajectory)

    wins = sum(1 for d in demonstrations if d["won"])
    print(f"Generated {n_episodes} demonstrations, {wins} wins ({wins/n_episodes*100:.1f}%)")

    return demonstrations
