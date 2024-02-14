import pygame
import random
import numpy as np
from src.envs.contracts import DeepEnv


class GridWorldEnv(DeepEnv):
    """
    Grid World environment
    The agent starts in the top left corner and must reach the bottom right corner.
    The agent can move up, down, left or right.
    If the agent reaches the bottom right corner, it receives a reward of 1.
    If the agent reaches the losing square, it receives a reward of -1.
    up = 0
    down = 1
    left = 2
    right = 3
    """

    def __init__(self, size=10):
        assert (size >= 3)
        self.OBS_SIZE = size * size
        self.ACTION_SIZE = 4
        self.size = size
        self.current_score = 0.0
        self.current_position = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.losing_square = None
        self.reset()
        self.nb_player = 1

    def generate_losing_square(self, is_random: bool = False):
        if is_random:
            losing_square = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            while losing_square in [(0, 0), self.goal]:
                losing_square = (
                    random.randint(0, self.size - 1),
                    random.randint(0, self.size - 1)
                )
        else:
            losing_square = (0, self.size - 1)
        return losing_square

    def reset(self):
        self.current_position = (0, 0)
        self.losing_square = self.generate_losing_square()
        self.current_score = 0.0

        return self.get_obs()

    def print(self):
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.current_position:
                    print(1, end=" ")
                elif (i, j) == self.goal:
                    print(2, end=" ")
                elif (i, j) == self.losing_square:
                    print(-1, end=" ")
                else:
                    print(0, end=" ")
            print()

    def available_actions_mask(self) -> np.ndarray:
        mask = np.zeros(self.ACTION_SIZE)

        if self.current_position[0] > 0:
            mask[0] = 1
        if self.current_position[0] < self.size - 1:
            mask[1] = 1
        if self.current_position[1] > 0:
            mask[2] = 1
        if self.current_position[1] < self.size - 1:
            mask[3] = 1

        return mask

    def available_actions(self) -> np.ndarray:
        return np.arange(self.ACTION_SIZE)[self.available_actions_mask().flatten() == 1]

    def sample(self):
        available_actions = np.arange(self.ACTION_SIZE)[self.available_actions_mask().flatten() == 1]
        return np.random.choice(available_actions)

    def get_score(self) -> float:
        return self.current_score

    def get_game_over(self) -> bool:
        return self.current_position == self.goal or self.current_position == self.losing_square

    def get_obs(self) -> np.ndarray:
        obs = np.zeros((self.size, self.size))
        obs[self.current_position] = 1
        obs[self.goal] = 2
        obs[self.losing_square] = -1
        return obs.reshape(self.OBS_SIZE)

    def step(self, action):
        next_position = None

        if action == 0:
            next_position = (self.current_position[0] - 1, self.current_position[1])
        elif action == 1:
            next_position = (self.current_position[0] + 1, self.current_position[1])
        elif action == 2:
            next_position = (self.current_position[0], self.current_position[1] - 1)
        elif action == 3:
            next_position = (self.current_position[0], self.current_position[1] + 1)

        if next_position is None:
            raise ValueError(f"Invalid action: {action}")

        if next_position[0] < 0 \
                or next_position[0] >= self.size \
                or next_position[1] < 0 \
                or next_position[1] >= self.size:
            next_position = self.current_position

        self.current_position = next_position

        if self.current_position == self.goal:
            self.current_score = 1.0
        elif self.current_position == self.losing_square:
            self.current_score = -1.0
        else:
            self.current_score = -0.1

        return self.get_obs(), self.current_score, self.get_game_over()

    def get_game_result_status(self):
        if self.get_game_over():
            if self.current_position == self.goal:
                return 1
            else:
                return 0
        else:
            return 0

    def step_play(self, action):
        return self.step(action)

    def clone_stochastic(self):
        return GridWorldEnv(self.size)

    def current_position_to_state(self):
        return self.current_position[0] * self.size + self.current_position[1]


class GridWorldGUI:
    def __init__(self):

        pygame.init()

        self.env = GridWorldEnv()
        self.cell_size = 100
        self.width = self.env.size * self.cell_size
        self.height = self.env.size * self.cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid World")

        self.font = pygame.font.SysFont(None, 30)
        self.game_over_font = pygame.font.SysFont(None, 50)

        # Define colors
        self.WHITE = (255, 255, 255)
        self.GREEN_FOREST = (34, 139, 34)
        self.GOAL_COLOR = (0, 0, 255, 128)
        self.LOSING_COLOR = (255, 0, 0, 128)

    def draw_grid(self):
        self.screen.fill(self.GREEN_FOREST)
        for i in range(self.env.size):
            for j in range(self.env.size):
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, self.WHITE, rect, 1)

    def draw_player(self):
        i, j = self.env.current_position
        x = j * self.cell_size + self.cell_size // 2
        y = i * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 5
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius)

    def draw_goal(self):
        i, j = self.env.goal
        rect = pygame.Rect(
            j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.GOAL_COLOR, rect)

    def draw_obstacle(self):
        if self.env.losing_square is not None:
            i, j = self.env.losing_square
            rect = pygame.Rect(
                j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, self.LOSING_COLOR, rect)

    def draw_text(self, text, x, y):
        surface = self.font.render(text, True, self.WHITE)
        self.screen.blit(surface, (x, y))

    def draw_game_over_screen(self, win):
        background = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        background.fill((30, 30, 30, 128))
        self.screen.blit(background, (0, 0))

        match win:
            case 1:
                self.game_over_text = self.game_over_font.render(
                    "You Win!", True, self.WHITE
                )
            case -1:
                self.game_over_text = self.game_over_font.render(
                    "You Lose.", True, self.WHITE
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
        pygame.draw.rect(self.screen, self.GREEN_FOREST, retry_button, 0)
        pygame.draw.rect(self.screen, self.WHITE, retry_button, 1)
        pygame.draw.rect(self.screen, self.GREEN_FOREST, quit_button, 0)
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
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_DOWN:
                        action = 1
                    elif event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3
                    else:
                        action = None

                    if action is not None:
                        self.env.step(action)
                        done = self.env.get_game_over()
                        if done:
                            self.draw_game_over_screen(self.env.get_score())
                            self.env.reset()

            self.screen.fill((0, 0, 0))
            self.draw_grid()
            self.draw_player()
            self.draw_goal()
            self.draw_obstacle()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
