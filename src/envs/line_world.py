import pygame
import numpy as np
from contracts import DeepEnv


class LineWorldEnv(DeepEnv):
    """
    Line World environment
    The agent starts in the leftmost square and must reach the rightmost square.
    The agent can move left or right.
    If the agent reaches the rightmost square, it receives a reward of 1.
    If the agent reaches the losing square, it receives a reward of -1.
    left = 0
    right = 1
    """
    def __init__(self, length=10):
        assert (length >= 3)
        self.OBS_SIZE = length
        self.ACTION_SIZE = 2
        self.length = length
        self.state = None
        self.goal = None
        self.losing_square = None
        self.current_score = 0.0
        self.reset()

    def reset(self):
        self.state = self.length // 2
        self.goal = self.length - 1
        self.losing_square = 0
    
    def print(self):
        # print an array of size length with a 1 at the current position
        # a 2 at the goal position
        # and a -1 at the losing position
        line = np.zeros(self.length)
        line[self.state] = 1
        line[self.goal] = 2
        line[self.losing_square] = -1
        print(line)

    def available_actions_mask(self) -> np.ndarray:
        return np.ones(self.ACTION_SIZE)
    
    def get_score(self) -> float:
        return self.current_score
    
    def get_game_over(self) -> bool:
        return self.state == self.goal or self.state == self.losing_square
    
    def get_obs(self) -> np.ndarray:
        line = np.zeros(self.length)
        line[self.state] = 1
        return line

    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.length - 1, self.state + 1)

        done = self.get_game_over()
        if done:
            self.current_score = 1.0 if self.state == self.goal else -1.0

    def clone_stochastic():
        pass


class LineWorldGUI:
    def __init__(self):
        pygame.init()
        self.env = LineWorldEnv()
        self.width = 600
        self.height = 400

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Line World")
        
        self.font = pygame.font.SysFont(None, 30)
        self.game_over_font = pygame.font.SysFont(None, 50)

        # Define colors
        self.WHITE = (255, 255, 255)
        self.GREEN_FOREST = (34, 139, 34)
        self.GOAL_COLOR = (0, 0, 255, 128)
        self.LOSING_COLOR = (255, 0, 0, 128)

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
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not running:
                        break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
                    else:
                        action = None
                    if action is not None:
                        self.env.step(action)
                        done = self.env.get_game_over()
                        if done:
                            self.draw_game_over_screen(self.env.get_score())
                            self.env.reset()

            self.screen.fill(self.GREEN_FOREST)
            for i in range(self.env.length):
                pygame.draw.rect(
                    self.screen,
                    self.GREEN_FOREST,
                    pygame.Rect(50 + i * 50, 200, 40, 40),
                    0,
                )
                pygame.draw.rect(
                    self.screen, self.WHITE, pygame.Rect(50 + i * 50, 200, 40, 40), 1
                )
            line = np.zeros(self.env.length)
            line[self.env.goal] = 2
            player_pos = (50 + self.env.state * 50 + 20, 200 + 20)
            pygame.draw.rect(
                self.screen,
                self.WHITE,
                pygame.Rect(player_pos[0] - 20, player_pos[1] - 20, 40, 40),
            )
            pygame.draw.rect(
                self.screen,
                self.GOAL_COLOR,
                pygame.Rect(50 + self.env.goal * 50, 200, 40, 40),
            )
            pygame.draw.rect(
                self.screen,
                self.LOSING_COLOR,
                pygame.Rect(50 + self.env.losing_square * 50, 200, 40, 40),
            )
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    LineWorldGUI().run()
