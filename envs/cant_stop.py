import numpy as np
import random
from contracts import DeepEnv


piece_to_int = {"RED": 1, "GREEN": 2}
checkpoint_to_int = {"RED": 5, "GREEN": 6}

#Return index of value in array
def array_idx(array, value):
    for i in range(len(array)):
        if array[i] == value:
            return i
    return -1


class Player():
    def __init__(self, color: str, random_player: bool, type: str) -> None:
        self.piece = color
        self.piece_int = piece_to_int[color]
        self.random_player = random_player
        self.checkpoint_int = checkpoint_to_int[color]
        self.nb_piece = 2
        self.type = type
        self.double_checkpoint_int = 7
        self.piece_with_checkpoint_int = 8
        self.piece_won = 0


class CantStop(DeepEnv):
    def __init__(self, is_headless: bool = False) -> None:
        self.OBS_SIZE = 83  # Nombre de case
        self.ACTION_SIZE = 83*2  # Nombre de case * 2 (continuer ou non)
        self.player = Player("RED", False, "player")
        self.opponent = Player("GREEN", False, "opponent")
        self.is_headless = is_headless
        self.reset()


    def reset(self):
        self.board = {
            2:     [0, 0, 0],
            3:     [0, 0, 0, 0, 0],
            4:     [0, 0, 0, 0, 0, 0, 0],
            5:     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            6:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            7:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            8:     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            9:     [0, 0, 0, 0, 0, 0, 0, 0, 0],
            10:    [0, 0, 0, 0, 0, 0, 0],
            11:    [0, 0, 0, 0, 0],
            12:    [0, 0, 0]
        }
        self.game_over = False
        self.score = 0

    def print(self):
        for idx, col in enumerate(self.board):
            print(str(idx+2), ":\t", self.board[col])

    def available_actions_mask(self, player: Player, opponent: Player) -> np.ndarray:
        mask = []

        for col in self.board:
            mask.append([0 for _ in range(len(self.board[col]))])
            
            idx_piece_player = array_idx(self.board[col], player.piece_int)
            idx_piece_opponent = array_idx(self.board[col], opponent.piece_int)

            idx_checkpoint_player = array_idx(self.board[col], player.checkpoint_int)
            # idx_checkpoint_opponent = array_idx(self.board[cols], opponent.piece_int)
            idx_checkpoint = array_idx(self.board[col], player.double_checkpoint_int)
            idx_piece_with_checkpoint = array_idx(self.board[col], player.piece_with_checkpoint_int)

            if idx_piece_opponent != (len(self.board[col]) - 1) and idx_piece_player != (len(self.board[col]) - 1):   # column not won by oponent or player
                if player.nb_piece > 0:                                                             # player has pieces to play

                    if idx_checkpoint_player != -1:                                               # player have a checkpoint on this column
                        mask[col - 2][idx_checkpoint_player] = 1

                    elif idx_checkpoint != -1:                                                      # player have a shared checkpoint on this column
                        mask[col - 2][idx_checkpoint] = 1

                    elif idx_checkpoint != -1:                                                      # player have a shared checkpoint on this column
                        mask[col - 2][idx_checkpoint] = 1

                    elif idx_piece_with_checkpoint != -1:                                           # player have a piece with a opponent checkpoint on this column
                        mask[col - 2][idx_piece_with_checkpoint + 1] = 1

                    elif idx_piece_player == -1 and idx_piece_opponent != 0:                          # player is not on the column or opponent is not on the start
                        mask[col - 2][0] = 1
                else:                                                                                   # player has no pieces to play
                    if idx_piece_player != -1 and idx_piece_player != len(self.board[col]) - 1:         # player can continue his progress on this column
                        mask[col - 2][idx_piece_player + 1] = 1            

                    elif idx_piece_with_checkpoint != -1:                                               # player have a piece with a opponent checkpoint on this column
                        mask[col - 2][idx_piece_with_checkpoint + 1] = 1

        mask.append([1, 1])                                                                         # Continue turn or not, always available

        return mask

    def get_score(self) -> float:
        return self.score

    def get_game_over(self) -> bool:
        self.player.piece_won = 0
        self.opponent.piece_won = 0

        for col in self.board:
            if self.board[col][-1] == self.player.piece_int:
                self.player.piece_won += 1
            if self.board[col][-1] == self.opponent.piece_int:
                self.opponent.piece_won += 1

        if self.player.piece_won >= 3:
            self.game_over = True
            self.score = 1
         
        elif self.opponent.piece_won >= 3: 
            self.game_over = True   
            self.score = -1 

    def get_obs(self) -> np.ndarray:
        return self.board

    def step(self, action, playable_idx, player: Player):

        self.board[action][playable_idx] = player.piece_int

        if playable_idx == 0:   # Move from the start
            player.nb_piece -= 1    
        elif self.board[action][playable_idx - 1] % 4 == player.checkpoint_int: # Move from a checkpoint
            player.nb_piece -= 1
        elif self.board[action][playable_idx - 1] % 4 == player.piece_int: # Delete older position
            self.board[action][playable_idx - 1] = 0

    def clone_stochastic(self) -> DeepEnv:
        pass

    def roll_dices(self):
        dice = []

        for _ in range(4):
            dice.append(random.randint(1, 6))
        return dice
    
    def get_available_combinations(self, dices, mask):
        combinations = set()

        for i in range(len(dices)):
            for j in range(i + 1, len(dices)):
                sum = dices[i] + dices[j]
                playable_idx = array_idx(mask[sum - 2], 1)

                if playable_idx != -1:
                    combinations.add((sum, playable_idx))

        return list(combinations)
    
    def save_checkpoint(self, player: Player):
        for col in self.board:
            idx_piece_player = array_idx(self.board[col], player.piece_int)
            idx_piece_with_checkpoint = array_idx(self.board[col], player.piece_with_checkpoint_int)

            if idx_piece_player != -1 and idx_piece_player != len(self.board[col]) - 1:
                self.board[col][idx_piece_player] = player.checkpoint_int
            elif idx_piece_with_checkpoint != -1 and idx_piece_with_checkpoint != len(self.board[col]) - 1:
                self.board[col][idx_piece_with_checkpoint] = player.double_checkpoint_int

            self.player.nb_piece = 3 - self.player.piece_won


    def play_turn(self, current_player: Player, current_opponent: Player):
            turn_over = False
            saved_board = self.board

            # Player turn
            while not turn_over:
                if not self.is_headless: print(f"{current_player.type} turn: {current_player.piece}({current_player.nb_piece})")
                dices = self.roll_dices()

                actions_mask = self.available_actions_mask(current_player, current_opponent)

                available_combinations = self.get_available_combinations(dices, actions_mask)

                if len(available_combinations) == 0:
                    turn_over = True
                    self.board = saved_board
                    break

                if current_player.random_player:
                    action = random.randint(0, len(available_combinations) - 1)
                    action = available_combinations[action]

                    self.step(action[0], action[1], current_player)

                    turn = ["y", "n"]

                    random_turn = random.randint(0, 1)

                    continue_turn = turn[random_turn]
                else:
                    # Input action

                    if not self.is_headless: print(f"Available combinations {str(dices)} ==> {str(available_combinations)}")

                    action = input("Choose the index of the action: ")

                    action = available_combinations[int(action)]

                    print("Chosen actions: ", action)

                    self.step(action[0], action[1], current_player)

                    continue_turn = input("Do you want to continue your turn? (y/n): ")

                if continue_turn == "n":
                    if not self.is_headless: print('End of turn\n')

                    self.save_checkpoint(current_player)

                    turn_over = True

                self.get_game_over()

                if self.game_over:
                    turn_over = True

                self.print() # Debug
                print(f"\nPieces:  {str(current_player.nb_piece)} - Piece won {str(current_player.piece_won)}") # Debug

    def play(self):
        while True:

            # Player turn
            self.play_turn(self.player, self.opponent)

            if self.game_over:
                break

            # Opponent turn
            self.play_turn(self.opponent, self.player)

            if self.game_over:
                break
        
        if not self.is_headless: print("Game over ! \n" + self.player.type + " won, your score: ", self.score)


game = CantStop()
game.play()
