import pygame
import numpy as np
import random
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


class TicTacToeGUI:
    def __init__(self, is_opponent_random: bool = True):
        pygame.init()

        self.is_opponent_random = is_opponent_random

        self.env = TicTacToeEnv()
        self.cell_size = 100
        self.width = 3 * self.cell_size
        self.height = 3 * self.cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tic Tac Toe")

        self.font = pygame.font.SysFont(None, 30)
        self.game_over_font = pygame.font.SysFont(None, 50)

        self.X_COLOR = (255, 0, 0)
        self.O_COLOR = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

    def draw_grid(self):
        self.screen.fill(self.WHITE)
        for i in range(1, 3):
            pygame.draw.line(self.screen, self.BLACK, (0, i * self.cell_size), (self.width, i * self.cell_size), 2)
            pygame.draw.line(self.screen, self.BLACK, (i * self.cell_size, 0), (i * self.cell_size, self.height), 2)

    def draw_symbols(self):
        for i in range(3):
            for j in range(3):
                x = j * self.cell_size + self.cell_size // 2
                y = i * self.cell_size + self.cell_size // 2

                if self.env.board[i, j] == 1:
                    pygame.draw.line(self.screen, self.X_COLOR, (x - 30, y - 30), (x + 30, y + 30), 2)
                    pygame.draw.line(self.screen, self.X_COLOR, (x + 30, y - 30), (x - 30, y + 30), 2)
                elif self.env.board[i, j] == -1:
                    pygame.draw.circle(self.screen, self.O_COLOR, (x, y), 30, 2)

    def draw_text(self, text, x, y):
        surface = self.font.render(text, True, self.BLACK)
        self.screen.blit(surface, (x, y))

    def draw_game_over_screen(self, win):
        background = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        background.fill((30, 30, 30, 128))
        self.screen.blit(background, (0, 0))

        match win:
            case 1.0:
                self.game_over_text = self.game_over_font.render(
                    "Cross Win !", True, self.WHITE
                )
            case -1.0:
                self.game_over_text = self.game_over_font.render(
                    "Rounds Win.", True, self.WHITE
                )
            case 0.0:
                self.game_over_text = self.game_over_font.render(
                    "It's a draw.", True, self.WHITE
                )
        self.screen.blit(
            self.game_over_text,
            (
                self.width // 2 - self.game_over_text.get_width() // 2,
                self.height // 2 - self.game_over_text.get_height() // 2,
            ),
        )

        retry_button = pygame.Rect(
            self.width // 2 - 100, self.height // 2 + 50, 100, 50
        )
        quit_button = pygame.Rect(self.width // 2 + 10, self.height // 2 + 50, 100, 50)
        pygame.draw.rect(self.screen, self.WHITE, retry_button, 0)
        pygame.draw.rect(self.screen, self.WHITE, retry_button, 1)
        pygame.draw.rect(self.screen, self.WHITE, quit_button, 0)
        pygame.draw.rect(self.screen, self.WHITE, quit_button, 1)
        self.draw_text("Retry", retry_button.x + 20, retry_button.y + 10)
        self.draw_text("Quit", quit_button.x + 20, quit_button.y + 10)

        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if retry_button.collidepoint(mouse_pos):
                        self.env.reset()
                        self.run()
                    elif quit_button.collidepoint(mouse_pos):
                        return

    def run(self):
        clock = pygame.time.Clock()
        done = False

        player_turn = np.random.choice([True, False])

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.MOUSEBUTTONDOWN and player_turn:
                    mouse_x, mouse_y = event.pos
                    col = mouse_x // self.cell_size
                    row = mouse_y // self.cell_size
                    action = row * 3 + col

                    try:
                        self.env.step_play(action)
                        player_turn = False
                    except ValueError as e:
                        print(e)

                elif not player_turn:
                    mask = self.env.available_actions_mask()
                    action = np.random.choice(np.arange(9), p=mask.flatten() / mask.sum())
                    try:
                        self.env.step_play(action)
                        player_turn = True
                    except ValueError as e:
                        print(e)

            self.screen.fill(self.WHITE)
            self.draw_grid()
            self.draw_symbols()

            if self.env.get_game_over():
                print(self.env.get_score())
                self.draw_game_over_screen(self.env.get_score())
                self.env.reset()
                player_turn = True

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    # gui = TicTacToeGUI()
    # gui.run()
    while True:
        env = TicTacToeEnv()
        env.print()
        done = False
        while not done:
            mask = env.available_actions_mask()
            action = input("Action: ")
            env.step_play(int(action))
            env.print()
            done = env.get_game_over()
            if done:
                print(env.get_score())
        print("-----")
