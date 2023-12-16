import pygame
import sys
import random


class CantStopGUI:
    def __init__(self):
        # Initialize pygame
        pygame.init()

        # Set the dimensions of the screen
        self.screen_width, self.screen_height = 800, 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # Set the box of buttons
        self.dice_rect_btn = pygame.Rect(self.screen_width - 120, 20, 100, 50)

        self.dice_rect_result = [
            pygame.Rect(self.screen_width - 120, 120, 100, 50),
            pygame.Rect(self.screen_width - 120, 120, 100, 50),
            pygame.Rect(self.screen_width - 120, 120, 100, 50),
            pygame.Rect(self.screen_width - 120, 120, 100, 50)
        ]

        self.continue_rect_btn = pygame.Rect(self.screen_width - 120, self.screen_height - 175, 100, 50)
        self.terminate_rect_btn = pygame.Rect(self.screen_width - 120, self.screen_height - 100, 100, 50)

        self.board_cases = []


        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (208, 19, 19)
        self.GREEN = (52, 170, 25)
        self.BLUE = (33, 85, 198)
        self.SQUARE_SIZE = 30
        self.padding = 10

        # Font for text
        self.font = pygame.font.SysFont(None, 24)

        # Board columns dictionary
        self.columns_dict = {
            2:  [0, 0, 0],
            3:  [0, 1, 0, 0, 0],
            4:  [0, 0, 0, 0, 0, 0, 0],
            5:  [0, 0, 2, 0, 0, 0, 0, 0, 0],
            6:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            7:  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            8:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            9:  [0, 0, 0, 2, 0, 0, 0, 0, 0],
            10: [0, 0, 0, 0, 0, 0, 0],
            11: [0, 0, 0, 0, 0],
            12: [0, 0, 0]
        }

        # Player pawns count example
        self.player_pawns_count = 3
        self.opponent_pawns_count = 2

    def draw_square(self, x, y, color=None):
        if color is None:
            color = self.BLACK
        pygame.draw.rect(self.screen, color, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE), 1)

    def draw_square_with_circle(self, x, y):
        self.draw_square(x, y)
        pygame.draw.circle(self.screen, self.BLACK, (x + self.SQUARE_SIZE // 2, y + self.SQUARE_SIZE // 2), 10)

    def draw_square_with_triangle(self, x, y):
        self.draw_square(x, y)
        pygame.draw.polygon(self.screen, self.BLACK, [(x + self.SQUARE_SIZE // 2, y + 5), 
                                                      (x + 5, y + self.SQUARE_SIZE - 5), 
                                                      (x + self.SQUARE_SIZE - 5, y + self.SQUARE_SIZE - 5)])

    def draw_board(self):
        # Calculate the total height and width of the pyramid
        total_height = len(self.columns_dict) * (self.SQUARE_SIZE + self.padding) - self.padding
        total_width = max(len(row) for row in self.columns_dict.values()) * (self.SQUARE_SIZE + self.padding) - self.padding

        # Calculate the starting x and y positions (centered on screen)
        start_x = (self.screen.get_height() // 2) - (total_height // 2) + 100
        start_y = total_width - self.padding

        # Draw the board based on the dictionary
        for i, (_, row) in enumerate(self.columns_dict.items()):
            # Calculate x position for the current column
            x = start_x + i * (self.SQUARE_SIZE + self.padding)
            
            y = start_y
            self.board_cases.append((pygame.Rect(x, y, self.SQUARE_SIZE, self.SQUARE_SIZE), i + 2, 0))
            
            for j, value in enumerate(row):

                if value == 0:
                    self.draw_square(x, y)
                elif value == 1:
                    self.draw_square_with_triangle(x, y)
                elif value == 2:
                    self.draw_square_with_circle(x, y)
                y -= self.SQUARE_SIZE + self.padding
                

    def draw_players(self):
        player_text = self.font.render('Player: RED', True, self.RED)
        opponent_text = self.font.render('Opponent: GREEN', True, self.GREEN)
        self.screen.blit(player_text, (20, 20))
        self.screen.blit(opponent_text, (20, 60))

    def roll_dice(self):
        # Rolls 4 six-sided dice and returns a list of their values
        return [random.randint(1, 6) for _ in range(4)]

    def draw_dice(self):
        # Placeholder for dice
        for i in range(4):
            pygame.draw.rect(self.screen, self.BLACK, (self.screen_height + (i * 50), 20, 40, 40), 2)

    def draw_dice(self, dice_values=None):
        # If dice_values is None, draw the roll button, else draw the dice results
        if dice_values is None:
            # Draw the roll dice button
            pygame.draw.rect(self.screen, self.BLACK, self.dice_rect_btn)
            text = self.font.render('Roll Dice', True, self.WHITE)
            self.screen.blit(text, text.get_rect(center=self.dice_rect_btn.center))
        else:
            # Draw the dice results
            for i, value in enumerate(dice_values):
                die_rect = pygame.Rect(self.screen_width - 120 + (i * (self.SQUARE_SIZE + 5)), 100, self.SQUARE_SIZE, self.SQUARE_SIZE)
                pygame.draw.rect(self.screen, self.WHITE, die_rect, 0)
                pygame.draw.rect(self.screen, self.BLACK, die_rect, 1)
                text = self.font.render(str(value), True, self.BLACK)
                self.screen.blit(text, text.get_rect(center=die_rect.center))

    def draw_remaining_pawns(self, player_pawns_count, opponent_pawns_count):
        remaining_pawns_text = self.font.render(f'Player Pawns: {player_pawns_count}', True, self.RED)
        self.screen.blit(remaining_pawns_text, (0, 550))
        remaining_pawns_text_2 = self.font.render(f'Opponent Pawns: {opponent_pawns_count}', True, self.GREEN)
        self.screen.blit(remaining_pawns_text_2, (0, 575))

    def draw_continue_button(self):
        # Draw the continue button
        pygame.draw.rect(self.screen, self.BLUE, self.continue_rect_btn)
        text = self.font.render('Continue', True, self.WHITE)
        self.screen.blit(text, text.get_rect(center=self.continue_rect_btn.center))

    def draw_terminate_button(self):
        # Draw the terminate button
        pygame.draw.rect(self.screen, self.BLUE, self.terminate_rect_btn)
        text = self.font.render('Terminate', True, self.WHITE)
        self.screen.blit(text, text.get_rect(center=self.terminate_rect_btn.center))

    def handle_events(self):
        # Handle events, including the roll dice button click
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                # Check if the roll dice button is clicked
                if self.dice_rect_btn.collidepoint(mouse_pos):
                    self.dice_values = self.roll_dice()
                    self.draw_dice(self.dice_values)
                    print("dice")
                
                # Check if the continue button is clicked
                if self.continue_rect_btn.collidepoint(mouse_pos):
                    print("continue")
                
                # Check if the terminate button is clicked
                if self.terminate_rect_btn.collidepoint(mouse_pos):
                    print("terminate")

                for rect in self.board_cases:
                    if rect[0].collidepoint(mouse_pos):
                        print("case:", rect[1], rect[2])


    def start_game(self):
        running = True
        while running:
            self.handle_events()

            self.screen.fill(self.WHITE)

            self.draw_board()
            self.draw_players()
            self.draw_dice()
            self.draw_continue_button()
            self.draw_terminate_button()
            self.draw_remaining_pawns(3, 2)

            pygame.display.flip()

        pygame.quit()
        sys.exit()

# Usage:
game_gui = CantStopGUI()
game_gui.start_game()