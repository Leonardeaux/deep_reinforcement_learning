import pygame
import itertools


class TicTacToeLogic:
    def __init__(self):
        self.board = [["", "", ""], ["", "", ""], ["", "", ""]]
        self.player = "X"
        
    def reset(self):
        self.board = [["", "", ""], ["", "", ""], ["", "", ""]]
        self.player = "X"

    def check_win(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != "":
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != "":
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != "":
            return True
        return self.board[0][2] == self.board[1][1] == self.board[2][0] != ""

    def play(self, i, j):
        if self.board[j][i] == "":
            self.board[j][i] = self.player
            if self.check_win():
                match self.player:
                    case "X":
                        return 1
                    case "O":
                        return -1
            else:
                self.player = "O" if self.player == "X" else "X"
        return None


class TicTacToeGUI:
    def __init__(self):
        pygame.init()
        
        self.env = TicTacToeLogic()
        self.height = 400
        self.width = 400

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tic Tac Toe")

        self.font = pygame.font.SysFont(None, 30)
        self.game_over_font = pygame.font.SysFont(None, 50)

        self.WHITE = (255, 255, 255)
        self.MIDNIGHT_BLUE = (25, 25, 112)

    def draw_board(self, board):
        self.screen.fill(self.MIDNIGHT_BLUE)
        for i in range(1, 3):
            pygame.draw.line(self.screen, self.WHITE, (0, i * 133), (400, i * 133), 5)
            pygame.draw.line(self.screen, self.WHITE, (i * 133, 0), (i * 133, 400), 5)
        for i, j in itertools.product(range(3), range(3)):
            if board[i][j] == "X":
                pygame.draw.line(
                    self.screen,
                    self.WHITE,
                    (j * 133 + 20, i * 133 + 20),
                    (j * 133 + 113, i * 133 + 113),
                    5,
                )
                pygame.draw.line(
                    self.screen,
                    self.WHITE,
                    (j * 133 + 113, i * 133 + 20),
                    (j * 133 + 20, i * 133 + 113),
                    5,
                )
            elif board[i][j] == "O":
                pygame.draw.circle(
                    self.screen, self.WHITE, (j * 133 + 66, i * 133 + 66), 60, 5
                )
                
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
        logic = TicTacToeLogic()

        self.draw_board(logic.board)
        pygame.display.update()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    i, j = x // 133, y // 133
                    if winner := logic.play(i, j):
                        self.draw_board(logic.board)
                        self.draw_game_over_screen(winner)
                        running = False
                    else:
                        self.draw_board(logic.board)
                        pygame.display.update()
                    if not running:
                        break

        pygame.quit()


if __name__ == "__main__":
    TicTacToeGUI().run()
