import numpy as np
import random
import pygame
from envs.contracts import DeepEnv

class TicTacToeEnv(DeepEnv):
    """
    Tic Tac Toe environment
    Players take turns placing their symbol (X or O) in an empty cell.
    The game ends when one player has three of their symbols in a line or the grid is full.
    X = 1, O = -1, empty = 0
    Player X = 0, Player O = 1
    """
    def __init__(self):
        # 3 puissance 9 états possibles
        self.OBS_SIZE = 3**9
        self.ACTION_SIZE = 3 * 3
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
        mask = np.array(np.zeros((3, 3)))
        
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    mask[i, j] = 1

        return mask
    
    def get_score(self) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == 0 else -1.0

    def get_game_over(self) -> bool:
        return self.done
    
    def get_obs(self) -> np.ndarray:
        return self.board.reshape(1, self.OBS_SIZE)

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

    def step_play(self, action):
        """Play a move in the game used by the player"""
        if self.done:
            raise ValueError("Game is over")
        if self.board[action // 3, action % 3] != 0:
            raise ValueError("Invalid move")
        
        self.board[action // 3, action % 3] = 1 if self.current_player == 0 else -1
        self.check_winner()
        self.current_player = 1 - self.current_player

    def step(self, action):
        """Play a move in the game used in the training loop"""
        if self.done:
            raise ValueError("Game is over")
        if self.board[action // 3, action % 3] != 0:
            raise ValueError("Invalid move")
        
        self.board[action // 3, action % 3] = 1 if self.current_player == 0 else -1
        self.check_winner()
        self.current_player = 1 - self.current_player

        random_action = random.choice(np.arange(9)[self.available_actions_mask().flatten() == 1])
        self.board[random_action // 3, random_action % 3] = 1 if self.current_player == 0 else -1
        self.check_winner()
        self.current_player = 1 - self.current_player

    def clone_stochastic(self):
        return TicTacToeEnv(self)
    
    def state_flatten(self):
        return self.board.flatten()
    
    def state_to_index(self, state):
        base3 = np.array([3**i for i in range(9)])
        flattened = state.flatten()
        flattened[flattened == -1] = 2  # Convertit -1 en 2 pour la représentation en base 3
        return int(np.dot(flattened, base3))
    
    def index_to_state(self, index):
        base3_representation = np.base_repr(index, base=3).zfill(9)  # Convertit en base 3 et remplit pour s'assurer qu'il a une longueur de 9
        state_values = np.array([int(digit) for digit in base3_representation])
        state_values[state_values == 2] = -1  # Convertit 2 en -1 pour obtenir les valeurs originales
        return state_values.reshape((3, 3))

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
                player_turn = np.random.choice([True, False])

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

class RandomAgent:
    def __init__(self, env: TicTacToeEnv):
        self.env = env

    def act(self, obs):
        mask = self.env.available_actions_mask()
        return random.choice(np.arange(9)[mask.flatten() == 1])
    
    def train(self):
        pass

    def test(self):
        pass



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