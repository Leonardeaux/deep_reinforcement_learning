import numpy as np
import random
import pygame
from contracts import DeepEnv

class TicTacToeEnv(DeepEnv):
    """
    Tic Tac Toe environment
    Players take turns placing their symbol (X or O) in an empty cell.
    The game ends when one player has three of their symbols in a line or the grid is full.
    X = 1, O = -1, empty = 0
    Player X = 0, Player O = 1
    """
    def __init__(self):
        self.OBS_SIZE = 3 * 3
        self.ACTION_SIZE = 9
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 0
        self.done = False
        self.winner = None

    def print(self):
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 1:
                    print("X", end=" ")
                elif self.board[i, j] == -1:
                    print("O", end=" ")
                else:
                    print("-", end=" ")
            print()

    def available_actions_mask(self) -> np.ndarray:
        return (self.board == 0).astype(int).flatten()
    
    def get_score(self) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == self.current_player else -1.0

    def get_game_over(self) -> bool:
        return self.done
    
    def get_obs(self) -> np.ndarray:
        return self.board.reshape(1, self.OBS_SIZE)

    def check_winner(self):
        # Check rows, columns and diagonals for a win
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3 or abs(self.board[:, i].sum()) == 3:
                self.done = True
                self.winner = self.current_player
                return
        if abs(np.diag(self.board).sum()) == 3 or abs(np.diag(np.fliplr(self.board)).sum()) == 3:
            self.done = True
            self.winner = self.current_player
            return

        # Check for draw
        if not np.any(self.board == 0):
            self.done = True
            self.winner = None

    def step(self, action):
        if self.done:
            raise ValueError("Game is over")
        if self.board[action // 3, action % 3] != 0:
            raise ValueError("Invalid move")
        
        self.board[action // 3, action % 3] = 1 if self.current_player == 0 else -1
        self.check_winner()
        self.current_player = 1 - self.current_player

    def clone_stochastic():
        pass

class TicTacToeGUI:
    def __init__(self):
        pygame.init()

        self.env = TicTacToeEnv()
        self.cell_size = 100
        self.width = 3 * self.cell_size
        self.height = 3 * self.cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tic Tac Toe")

        self.font = pygame.font.SysFont(None, 30)

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

    def run(self):
        clock = pygame.time.Clock()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    col = mouse_x // self.cell_size
                    row = mouse_y // self.cell_size
                    action = row * 3 + col

                    try:
                        self.env.step(action)
                    except ValueError as e:
                        print(e)

            self.screen.fill(self.WHITE)
            self.draw_grid()
            self.draw_symbols()

            if self.env.get_game_over():
                winner_text = "Player X wins!" if self.env.get_score() == 1 else "Player O wins!" if self.env.get_score() == -1 else "It's a draw!"
                self.draw_text(winner_text, 10, self.height // 2 - 30)
                self.draw_text("Click to play again", 10, self.height // 2 + 10)

                if pygame.mouse.get_pressed()[0]:
                    self.env.reset()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    gui = TicTacToeGUI()
    gui.run()