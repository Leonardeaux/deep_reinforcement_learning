import pygame
import random


class GridWorldLogic:
    def __init__(self):
        self.size = 5
        self.current_position = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.obstacle = None
        self.reset()

    def reset(self):
        self.current_position = (0, 0)
        self.obstacle = self.generate_obstacle()

    def generate_obstacle(self):
        obstacle = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        while obstacle in [(0, 0), self.goal]:
            obstacle = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )
        return obstacle

    def step(self, action):
        if action == "up":
            next_position = (self.current_position[0] - 1, self.current_position[1])
        elif action == "down":
            next_position = (self.current_position[0] + 1, self.current_position[1])
        elif action == "left":
            next_position = (self.current_position[0], self.current_position[1] - 1)
        elif action == "right":
            next_position = (self.current_position[0], self.current_position[1] + 1)

        if (
            next_position[0] < 0
            or next_position[0] >= self.size
            or next_position[1] < 0
            or next_position[1] >= self.size
        ):
            next_position = self.current_position

        self.current_position = next_position

        if self.current_position == self.goal:
            reward = 1
            done = True
        elif self.current_position == self.obstacle:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        return self.current_position, reward, done, {}


class GridWorldGUI:
    def __init__(self):
        pygame.init()

        self.env = GridWorldLogic()
        self.cell_size = 100
        self.width = self.env.size * self.cell_size
        self.height = self.env.size * self.cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid World")

        self.font = pygame.font.SysFont(None, 30)
        self.game_over_font = pygame.font.SysFont(None, 50)

        # Define colors
        self.WHITE = (255, 255, 255)
        self.MIDNIGHT_BLUE = (25, 25, 112)

    def draw_grid(self):
        self.screen.fill(self.MIDNIGHT_BLUE)
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
        pygame.draw.rect(self.screen, (0, 255, 0, 128), rect)

    def draw_obstacle(self):
        if self.env.obstacle is not None:
            i, j = self.env.obstacle
            rect = pygame.Rect(
                j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(self.screen, (255, 0, 0, 128), rect)

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
        clock = pygame.time.Clock()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = "up"
                    elif event.key == pygame.K_DOWN:
                        action = "down"
                    elif event.key == pygame.K_LEFT:
                        action = "left"
                    elif event.key == pygame.K_RIGHT:
                        action = "right"
                    else:
                        action = None

                    if action is not None:
                        state, reward, done, _ = self.env.step(action)

            self.screen.fill((0, 0, 0))
            self.draw_grid()
            self.draw_player()
            self.draw_goal()
            self.draw_obstacle()
            pygame.display.flip()
            clock.tick(60)

        if reward != 0:
            self.draw_game_over_screen(reward)

        pygame.quit()


if __name__ == "__main__":
    gui = GridWorldGUI().run()
