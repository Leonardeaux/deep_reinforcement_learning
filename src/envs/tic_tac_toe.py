import numpy as np
from src.envs.contracts import DeepEnv


class TicTacToeEnv(DeepEnv):
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 0
        self.done = False
        self.winner = None
        self.OBS_SIZE = 3 * 3
        self.ACTION_SIZE = 3 * 3
        self.nb_player = 2

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 0
        self.done = False
        self.winner = None

        return self.board

    def print(self):
        symbols = {0: "-", 1: "X", -1: "O"}
        for row in self.board:
            print(' '.join(symbols[int(cell)] for cell in row))

    def available_actions_mask(self) -> np.ndarray:
        return (self.board == 0).astype(int)

    def available_actions(self) -> np.ndarray:
        return np.arange(self.ACTION_SIZE)[self.available_actions_mask().flatten() == 1]

    def sample(self):
        """Return a random action index from the available action space."""
        available_actions = np.arange(self.ACTION_SIZE)[self.available_actions_mask().flatten() == 1]
        return np.random.choice(available_actions)

    def get_game_result_status(self):
        if self.done:
            if self.get_score() == 1:
                return 1
            if self.get_score() == -1:
                return 0
            return 0.5

        raise ValueError("Game is not over")

    def get_score(self):
        if not self.done:
            return 0  # Not done
        if self.winner is None:
            return 0.5  # Draw
        return 1 if self.winner == 0 else -1  # X wins if the winner is 0, O wins if the winner is 1

    def get_game_over(self) -> bool:
        return self.done

    def get_obs(self) -> np.ndarray:
        return self.board.reshape(self.OBS_SIZE)

    def check_winner(self):
        for i in range(3):
            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                self.winner = 0
                self.done = True
                return
            elif np.all(self.board[i, :] == -1) or np.all(self.board[:, i] == -1):
                self.winner = 1
                self.done = True
                return

        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
            self.winner = 0
            self.done = True
            return
        elif np.all(np.diag(self.board) == -1) or np.all(np.diag(np.fliplr(self.board)) == -1):
            self.winner = 1
            self.done = True
            return

        if np.all(self.board != 0):
            self.winner = None
            self.done = True
            return

    def step(self, action):
        if self.done:
            raise ValueError("Game is over")

        row, col = divmod(action, 3)

        if self.board[row, col] != 0:
            raise ValueError("Invalid move")

        self.board[row, col] = 1 if self.current_player == 0 else -1

        self.check_winner()

        if self.done:
            return self.board.copy(), self.get_score(), True

        self.play_random_opponent()

        self.check_winner()

        reward = self.get_score()

        return self.board.copy(), reward, self.done

    def step_play(self, action):
        """Play a move in the game used by the player"""
        if self.done:
            raise ValueError("Game is over")
        if self.board[action // 3, action % 3] != 0:
            raise ValueError("Invalid move")

        self.board[action // 3, action % 3] = 1 if self.current_player == 0 else -1
        self.check_winner()
        self.current_player = 1 - self.current_player

        return self.board.copy(), self.get_score(), self.done

    def play_random_opponent(self):
        available = np.argwhere(self.board == 0)
        if len(available) == 0:
            return

        move = available[np.random.choice(len(available))]
        self.board[move[0], move[1]] = -1

    def clone_stochastic(self):
        return TicTacToeEnv(self)
