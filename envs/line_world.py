import pygame
import numpy as np


class LineWorldLogic:
    def __init__(self, length=10):
        self.length = length
        self.state = None
        self.goal = None
        self.reset()

    def reset(self):
        self.state = np.random.randint(self.length)
        self.goal = np.random.choice([i for i in range(self.length) if i != self.state])
        return self.state

    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.length - 1, self.state + 1)

        done = self.state == self.goal
        reward = 1 if done else 0
        return self.state, reward, done

    def get_state(self):
        return self.state


class LineWorldGUI:
    def __init__(self):
        pygame.init()
        self.env = LineWorldLogic()
        self.width = 600
        self.height = 400

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Line World")
        
        self.font = pygame.font.SysFont(None, 30)
        self.game_over_font = pygame.font.SysFont(None, 50)

        self.WHITE = (255, 255, 255)
        self.MIDNIGHT_BLUE = (25, 25, 112)
        self.GREEN = (0, 255, 0)

    def draw_text(self, text, x, y):
        surface = self.font.render(text, True, self.WHITE)
        self.screen.blit(surface, (x, y))

    def draw_game_over_screen(self, win):
        background = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        background.fill((30, 30, 30, 128))
        self.screen.blit(background, (0, 0))

        self.game_over_text = self.game_over_font.render("You Win!", True, self.WHITE)
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
        pygame.draw.rect(self.screen, self.MIDNIGHT_BLUE, retry_button, 0)
        pygame.draw.rect(self.screen, self.WHITE, retry_button, 1)
        pygame.draw.rect(self.screen, self.MIDNIGHT_BLUE, quit_button, 0)
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
                        state, reward, done = self.env.step(action)
                        if done:
                            self.draw_game_over_screen(1)
                            self.env.reset()

            self.screen.fill(self.MIDNIGHT_BLUE)
            for i in range(self.env.length):
                pygame.draw.rect(
                    self.screen,
                    self.MIDNIGHT_BLUE,
                    pygame.Rect(50 + i * 50, 200, 40, 40),
                    0,
                )
                pygame.draw.rect(
                    self.screen, self.WHITE, pygame.Rect(50 + i * 50, 200, 40, 40), 1
                )
            line = np.zeros(self.env.length)
            line[self.env.goal] = 2
            player_pos = (50 + self.env.get_state() * 50 + 20, 200 + 20)
            pygame.draw.rect(
                self.screen,
                self.WHITE,
                pygame.Rect(player_pos[0] - 20, player_pos[1] - 20, 40, 40),
            )
            pygame.draw.rect(
                self.screen,
                self.GREEN,
                pygame.Rect(50 + self.env.goal * 50, 200, 40, 40),
            )
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    LineWorldGUI().run()
