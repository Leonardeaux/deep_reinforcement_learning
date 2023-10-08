import numpy as np
import random
from contracts import DeepEnv


#Return index of value in array
def array_idx(array, value):
    for i in range(len(array)):
        if array[i] % 4 == value: # If value is in array we did a modulo to have the color and the checkpoint of the player
            return i
    return -1


class Player():
    def __init__(self, color: str, random_player: bool, type: str) -> None:
        piece_to_int = {"RED": 1, "GREEN": 2, "BLUE": 3, "YELLOW": 4}
        checkpoint = {"RED": 5, "GREEN": 6, "BLUE": 7, "YELLOW": 8}

        self.piece = color
        self.piece_int = piece_to_int[color]
        self.random_player = random_player
        self.checkpoint = checkpoint[color]
        self.nb_piece = 3
        self.type = type


class CantStop(DeepEnv):
    def __init__(self) -> None:
        self.OBS_SIZE = 83  # Nombre de case
        self.ACTION_SIZE = 83*2  # Nombre de case * 2 (continuer ou non)
        self.player = Player("RED", True, "player")
        self.opponent = Player("GREEN", True, "opponent")
        self.reset()


    def reset(self):
        self.board = {
            2:     [0, 0, 0],
            3:     [0, 0, 0, 0, 0],
            4:     [0, 0, 0, 0, 0, 0, 0],
            5:     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            6:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            7:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            8:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            9:     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            10:    [0, 0, 0, 0, 0, 0, 0],
            11:    [0, 0, 0, 0, 0],
            12:    [0, 0, 0]
        }

        self.game_over = False
        self.score = 0

    def print(self):
        print(self.board)

    def available_actions_mask(self, player: Player, opponent: Player) -> np.ndarray:
        mask = []

        # Available actions for each column of the board
        for cols in self.board:
            mask.append([0 for _ in range(len(self.board[cols]))])
            idx_player = array_idx(self.board[cols], player.piece_int)
            idx_opponent = array_idx(self.board[cols], opponent.piece_int)

            if idx_opponent != len(self.board[cols]) - 1:   # column not won by oponent
                if player.nb_piece > 0: # player has pieces to play

                    # If player is not on the column
                    if idx_player == -1:
                        mask[cols - 2][0] = 1

                    elif idx_player != len(self.board[cols]) - 1:
                        mask[cols - 2][idx_player + 1] = 1
                else:

                    if idx_player != -1:
                        mask[cols - 2][idx_player + 1] = 1

        mask.append([1, 1]) # Continue turn or not, always available

        return mask


    def get_score(self) -> float:
        return self.score

    def get_game_over(self) -> bool:
        player_cpt = 0
        opponent_cpt = 0

        for col in self.board:
            if self.board[col][-1] % 4 == self.player.piece_int:
                player_cpt += 1
            if self.board[col][-1] % 4 == self.opponent.piece_int:
                opponent_cpt += 1

        if player_cpt >= 3:
            self.game_over = True
            self.score = 1
         
        elif opponent_cpt >= 3: 
            self.game_over = True   
            self.score = -1 

        return self.game_over

    def get_obs(self) -> np.ndarray:
        return self.board

    def step(self, action, playable_idx, player: Player):

        self.board[action][playable_idx] = player.piece_int
        if playable_idx == 0:
            player.nb_piece -= 1

    def clone_stochastic(self) -> DeepEnv:
        pass

    def roll_dices(self):
        dice = []

        for _ in range(4):
            dice.append(random.randint(1, 6))
        return dice
    
    def get_available_combinations(self, dices, mask):
        combinations = []

        for i in range(len(dices)):

            for j in range(i + 1, len(dices)):
                sum = dices[i] + dices[j]
                playable_idx = array_idx(mask[sum - 2], 1)

                if playable_idx != -1:
                    combinations.append([sum, playable_idx])

        return combinations

    def play_turn(self, player: Player, opponent: Player):
            turn_over = False
            saved_board = self.board

            # Player turn
            while not turn_over:
                # print(player.type + " turn: ", player.piece)
                dices = self.roll_dices()

                actions_mask = self.available_actions_mask(player, opponent)

                available_combinations = self.get_available_combinations(dices, actions_mask)

                if len(available_combinations) == 0:
                    turn_over = True
                    self.board = saved_board
                    break

                if player.random_player:
                    action = random.randint(0, len(available_combinations) - 1)
                    action = available_combinations[action]

                    self.step(action[0], action[1], player)

                    turn = ["y", "n"]

                    random_turn = random.randint(0, 1)

                    continue_turn = turn[random_turn]

                else:
                    # Input action
                    print("Available combinations: ", available_combinations)
                    action = input("Choose the index of the action: ")
                    action = available_combinations[int(action) - 1]
                
                    self.step(action[0], action[1], player)

                    continue_turn = input("Do you want to continue your turn? (y/n): ")

                # if continue_turn == "n":
                    # print('End of turn\n')
                turn_over = True

    def play(self):
        while not self.game_over:

            # Player turn
            self.play_turn(self.player, self.opponent)

            # Opponent turn
            self.play_turn(self.opponent, self.player)

            self.get_game_over()
        
        print("Game over, " + self.player.type + " score: ", self.score)


game = CantStop()
game.play()
