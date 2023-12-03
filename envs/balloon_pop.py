import random

class BallonPop:
    def __init__(self):
        self.colors = ['RED', 'YELLOW', 'BLUE']
        self.symbols = ['MOON', 'STAR', 'DIAMOND']
        self.score_sheet = {
            'YELLOW': [0, 3, 7, 11, 15, 3],
            'BLUE': [1, 3, 5, 7, 9, 12, 8],
            'RED': [0, 0, 0, 2, 4, 6, 8, 10, 14, 6],
            'STAR': [1, 2, 3, 5, 7, 10, 13, 16, 4],
            'MOON': [2, 3, 4, 5, 7, 9, 12, 5],
            'DIAMOND': [1, 3, 6, 10, 13, 7]
        }
        self.current_scores = {key: 0 for key in self.score_sheet}
        self.final_scores = []
        self.game_over = False
        self.dice_rolls = []
        self.reset()

    def reset(self):
        self.current_scores = {key: 0 for key in self.score_sheet}
        self.final_scores = []
        self.game_over = False
        self.dice_rolls = []

    def print(self):
        print("\nCurrent score sheet:")
        for key, values in self.score_sheet.items():
            current_position = self.current_scores[key]
            score_line = ' '.join(f"{value:2d}" for value in values)
            progress_line = '|' * current_position + '.' * (len(values) - current_position)
            print(f"{key}: {progress_line} ({current_position}/{len(values)}) - Scores: {score_line}")

    def get_score(self):
        return sum(self.final_scores)

    def get_game_over(self):
        return self.game_over

    def get_obs(self):
        return self.current_scores

    def step(self, kept_dice):
        # Update scores based on kept dice
        for color, symbol in kept_dice:
            if self.current_scores[color] < len(self.score_sheet[color]) - 1:
                self.current_scores[color] += 1
            if self.current_scores[symbol] < len(self.score_sheet[symbol]) - 1:
                self.current_scores[symbol] += 1

        # Check for completed columns and score them
        for key in self.score_sheet.keys():
            if self.current_scores[key] == len(self.score_sheet[key]) - 1 and key not in self.final_scores:
                self.final_scores.append(sum(self.score_sheet[key]))
                self.current_scores[key] += 1  # Mark as completed

    def roll_dices(self):
        return [random.choice(self.colors), random.choice(self.symbols)]

    def get_available_combinations(self):
        # Provide options for dice to keep or reroll
        options = [f"Dé {i+1}: {self.dice_rolls[i]}" for i in range(len(self.dice_rolls))]
        return options

    def play_turn(self):
        self.dice_rolls = [self.roll_dices() for _ in range(3)]  # Initial roll of 3 dice
        num_rolls = 0
        while num_rolls < 2:
            self.print()
            print("Dés actuels:", self.dice_rolls)
            print("Options:", self.get_available_combinations())

            reroll_indices = input("Entrez les indices des dés à relancer (séparés par des espaces) ou appuyez sur Entrée pour conserver tous les dés: ")
            if reroll_indices:
                reroll_indices = [int(i) - 1 for i in reroll_indices.split()]
                new_dice_rolls = [self.dice_rolls[i] for i in range(len(self.dice_rolls)) if i not in reroll_indices]
                new_dice_rolls += [self.roll_dices() for _ in range(len(reroll_indices) + 1)]  # Relancer les dés sélectionnés et ajouter un dé supplémentaire
                self.dice_rolls = new_dice_rolls
            num_rolls += 1

        self.step(self.dice_rolls)  # Mise à jour des scores
        print("Dés finaux pour ce tour:", self.dice_rolls)

    def play(self):
        while not self.game_over:
            self.play_turn()
            if len(self.final_scores) >= 3:
                self.game_over = True
        print("Game over, final score: ", self.get_score())

# Exemple d'utilisation
game = BallonPop()
game.play()
