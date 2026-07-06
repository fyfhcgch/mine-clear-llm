import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, rows=9, cols=9, n_mines=10, render_mode=None, action_mode="reveal", max_steps=None):
        super().__init__()
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        if n_mines < 0 or n_mines >= rows * cols:
            raise ValueError("n_mines must be in [0, rows * cols)")
        if action_mode not in {"reveal", "reveal_flag"}:
            raise ValueError("action_mode must be 'reveal' or 'reveal_flag'")

        self.rows = rows
        self.cols = cols
        self.n_mines = n_mines
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.n_cells = rows * cols
        self.max_steps = max_steps or self.n_cells * (4 if action_mode == "reveal_flag" else 2)

        # reveal mode: action = cell to reveal.
        # reveal_flag mode: [0, n_cells) reveal, [n_cells, 2*n_cells) toggle flag.
        self.action_space = spaces.Discrete(self.n_cells * (2 if action_mode == "reveal_flag" else 1))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, rows, cols), dtype=np.float32
        )

        self._init_board()

    def _init_board(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.revealed = np.zeros((self.rows, self.cols), dtype=bool)
        self.flagged = np.zeros((self.rows, self.cols), dtype=bool)
        self.game_over = False
        self.game_won = False
        self.done = False
        self.first_move = True
        self.steps = 0
        self._place_mines()
        self._calc_numbers()

    def _place_mines(self):
        positions = self.np_random.choice(
            self.rows * self.cols, self.n_mines, replace=False
        )
        for pos in positions:
            r, c = divmod(int(pos), self.cols)
            self.board[r, c] = -1

    def _calc_numbers(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == -1:
                    continue
                count = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self.board[nr, nc] == -1:
                                count += 1
                self.board[r, c] = count

    def _flood_fill(self, r, c):
        stack = [(r, c)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                continue
            if self.revealed[r, c] or self.flagged[r, c]:
                continue
            self.revealed[r, c] = True
            if self.board[r, c] == 0:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        stack.append((r + dr, c + dc))

    def _ensure_safe_first_move(self, r, c):
        if self.board[r, c] != -1:
            return
        self.board[r, c] = 0
        while True:
            pos = int(self.np_random.integers(self.rows * self.cols))
            nr, nc = divmod(pos, self.cols)
            if self.board[nr, nc] != -1 and (nr, nc) != (r, c):
                self.board[nr, nc] = -1
                break
        self._calc_numbers()

    def _get_obs(self):
        obs = np.zeros((5, self.rows, self.cols), dtype=np.float32)
        obs[0] = self.revealed.astype(np.float32)
        obs[1] = self.flagged.astype(np.float32)
        obs[2] = self._number_layer()
        obs[3] = self._unrevealed_neighbors_layer()
        obs[4] = self._boundary_layer()
        return obs

    def _number_layer(self):
        layer = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.revealed[r, c] and self.board[r, c] > 0:
                    layer[r, c] = self.board[r, c] / 8.0
        return layer

    def _unrevealed_neighbors_layer(self):
        layer = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.revealed[r, c] and self.board[r, c] > 0:
                    count = 0
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                                if not self.revealed[nr, nc]:
                                    count += 1
                    layer[r, c] = count / 8.0
        return layer

    def _boundary_layer(self):
        layer = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.revealed[r, c] and not self.flagged[r, c]:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                                if self.revealed[nr, nc]:
                                    layer[r, c] = 1.0
                                    break
                        if layer[r, c] > 0:
                            break
        return layer

    def valid_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                if not self.revealed[r, c] and not self.flagged[r, c]:
                    mask[idx] = True
                if self.action_mode == "reveal_flag" and not self.revealed[r, c]:
                    mask[self.n_cells + idx] = True
        return mask

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        self.steps += 1
        truncated = self.steps >= self.max_steps

        action = int(action)
        if action < 0 or action >= self.action_space.n:
            return self._get_obs(), -1.0, False, truncated, {"invalid_action": True}

        if self.action_mode == "reveal_flag" and action >= self.n_cells:
            idx = action - self.n_cells
            r, c = divmod(idx, self.cols)
            if self.revealed[r, c]:
                return self._get_obs(), -1.0, False, truncated, {"invalid_action": True}
            self.flagged[r, c] = not self.flagged[r, c]
            return self._get_obs(), -0.05, False, truncated, {"flag_toggled": True}

        r, c = divmod(action, self.cols)
        reward = 0.0

        if self.first_move:
            self._ensure_safe_first_move(r, c)
            self.first_move = False

        if self.revealed[r, c] or self.flagged[r, c]:
            reward = -1.0
            return self._get_obs(), reward, False, truncated, {"invalid_action": True}

        if self.board[r, c] == -1:
            self.game_over = True
            self.done = True
            total_safe = self.rows * self.cols - self.n_mines
            progress = np.sum(self.revealed) / total_safe
            reward = -10.0 + progress * 5.0
            return self._get_obs(), reward, True, False, {"game_over": True}

        cells_before = np.sum(self.revealed)
        self._flood_fill(r, c)
        cells_after = np.sum(self.revealed)
        new_revealed = cells_after - cells_before

        total_safe = self.rows * self.cols - self.n_mines
        progress = cells_after / total_safe
        reward = 1.0 + new_revealed * 1.0 + progress * 2.0

        if cells_after >= total_safe:
            self.game_won = True
            self.game_over = True
            self.done = True
            reward = 100.0
            return self._get_obs(), reward, True, False, {"game_won": True}

        return self._get_obs(), reward, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_board()
        return self._get_obs(), {}

    def render(self):
        if self.render_mode != "human":
            return
        symbols = {0: ".", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", -1: "*"}
        print("\n  " + " ".join(str(i) for i in range(self.cols)))
        for r in range(self.rows):
            row = f"{r} "
            for c in range(self.cols):
                if self.game_over and self.board[r, c] == -1:
                    row += "* "
                elif self.flagged[r, c]:
                    row += "F "
                elif self.revealed[r, c]:
                    row += symbols[self.board[r, c]] + " "
                else:
                    row += "# "
            print(row)
        print()
