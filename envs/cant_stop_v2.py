import random

class CantStopLogic:
    def __init__(self):
        self.players = [1, 2]  # Replace with actual player names or IDs
        self.columns = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
        self.bonzes = {col: [] for col in self.columns}
        self.reserve_bonzes = {player: 3 for player in self.players}
        self.player_positions = {player: {col: 0 for col in self.columns} for player in self.players}
        self.current_player = random.choice(self.players)

    def roll_dice(self):
        return random.randint(1, 6), random.randint(1, 6)

    def valid_moves(self, dice1, dice2):
        possible_moves = []
        for col1 in self.columns:
            possible_moves.extend(
                (col1, col2)
                for col2 in self.columns
                if col1 != col2 and (col1 + col2 == dice1 + dice2)
            )
        return possible_moves

    def make_move(self, col1, col2):
        if (col1, col2) not in self.valid_moves(*self.roll_dice()):
            return False
        if len(self.bonzes[col1]) <= 0:
            self.bonzes[col1].append(self.current_player)
        elif self.current_player in self.bonzes[col1]:
            index = self.bonzes[col1].index(self.current_player)
            if index < len(self.bonzes[col1]) - 1:
                self.bonzes[col1][index + 1] = self.current_player

        if len(self.bonzes[col2]) <= 0:
            self.bonzes[col2].append(self.current_player)
        elif self.current_player in self.bonzes[col2]:
            index = self.bonzes[col2].index(self.current_player)
            if index < len(self.bonzes[col2]) - 1:
                self.bonzes[col2][index + 1] = self.current_player

        self.player_positions[self.current_player][col1] += 1
        self.player_positions[self.current_player][col2] += 1

        if all(len(self.bonzes[col]) >= 3 for col in self.columns):
            return self.check_winner()

        return True

    def check_winner(self):
        return next(
            (
                f"Player {self.current_player} wins the column {col}!"
                for col in self.columns
                if len(self.bonzes[col]) >= 3
                and self.current_player in self.bonzes[col]
            ),
            None,
        )

    def switch_player(self):
        index = self.players.index(self.current_player)
        self.current_player = self.players[(index + 1) % len(self.players)]

    def play_turn(self):
        print(f"Current player: Player {self.current_player}")
        dice1, dice2 = self.roll_dice()
        print(f"Dice rolled: {dice1}, {dice2}")

        possible_moves = self.valid_moves(dice1, dice2)
        print(f"Possible moves: {possible_moves}")

        if possible_moves:
            valid_input = False
            while not valid_input:
                try:
                    col1, col2 = map(int, input(f"Enter columns to choose (e.g., {possible_moves[0]}): ").split())
                    if (col1, col2) in possible_moves:
                        valid_input = True
                    else:
                        print("Invalid input. Please try again.")
                except ValueError:
                    print("Invalid input format. Please enter two integers separated by space.")

            print(f"Player {self.current_player} chooses columns {col1} and {col2}")

            if move_result := self.make_move(col1, col2):
                print("Move successful!\n")
        else:
            print("No valid moves. Skipping turn.\n")

        self.switch_player()


CantStopLogic().play_turn()