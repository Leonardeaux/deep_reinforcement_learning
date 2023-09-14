import itertools
import pygame

class TicTacToe:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Set up the window
        self.WINDOW_SIZE = (300, 300)
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("Tic Tac Toe")

        # Set up the board
        self.board = [['', '', ''], ['', '', ''], ['', '', '']]
        self.player = 'X'

        # Set up the font
        self.font = pygame.font.SysFont('Arial', 50)

        # Set up the colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)

    # Draw the board
    def draw_board(self):
        self.screen.fill(self.WHITE)
        for i, j in itertools.product(range(3), range(3)):
            pygame.draw.rect(self.screen, self.BLACK, (i*100, j*100, 100, 100), 3)
            text = self.font.render(self.board[j][i], True, self.BLACK)
            self.screen.blit(text, (i*100+35, j*100+20))

    # Check for a win
    def check_win(self):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != '':
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != '':
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '':
            return True
        return self.board[0][2] == self.board[1][1] == self.board[2][0] != ''

    # Main game loop
    def play(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    i, j = x//100, y//100
                    if self.board[j][i] == '':
                        self.board[j][i] = self.player
                        if self.check_win():
                            print(f'{self.player} wins!')
                            running = False
                        self.player = 'O' if self.player == 'X' else 'X'
            self.draw_board()
            pygame.display.update()

        # Quit Pygame
        pygame.quit()
        
if __name__ == '__main__':
    TicTacToe().play()