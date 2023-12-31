import numpy as np
import random

from numpy import ndarray
from contracts import DeepEnv
from Player import Player


#Return index of value in array
def array_idx(array, value):
    for i in range(len(array)):
        if array[i] == value:
            return i
    return -1

# CONSTANTS
PLAYER_PAWNS = 1
OPPONENT_PAWNS = 2
CHECKPOINT_PLAYER = 3
CHECKPOINT_OPPONENT = 4
CHECKPOINT_BOTH = 5
CHECKPOINT_PAWNS_CURRENT_PLAYER = 6     # Current Player have a pawn on a checkpoint


class CantStopEnv(DeepEnv):
    def __init__(self, is_headless: bool = False):
        self.is_headless = is_headless
        self.OBS_SIZE = 83 * 2
        self.ACTION_SIZE = 83

        self.reset()


    def reset(self):
        self.board = [
            [0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0]
        ]

        self.current_player = 0 # 0 = player, 1 = opponent
        self.score = 0
        self.game_over = False

        self.player_is_random = False
        self.opponent_is_random = False

        self.player_pawns_won = 0
        self.opponent_pawns_won = 0

        self.player_current_pawns = 3
        self.opponent_current_pawns = 3


    def print_mask(self, mask):
        print("Mask: ")
        for idx, col in enumerate(mask):
            print(str(idx + 2), ":\t", col)


    def get_playable_index(self, action: int) -> int:
        pawn_to_search = 1 if self.current_player == 0 else 2
        checkpoint_to_search = 3 if self.current_player == 0 else 4

        for idx, value in enumerate(self.board[action]):
            if value == pawn_to_search \
            or value == checkpoint_to_search \
            or value == CHECKPOINT_PAWNS_CURRENT_PLAYER \
            or value == CHECKPOINT_BOTH:
                return idx
        return -1
    

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
    

    def save_checkpoint(self):

        if self.current_player == 0:
            for col in self.board:
                idx_piece_with_checkpoint = array_idx(col, CHECKPOINT_PAWNS_CURRENT_PLAYER)
                idx_piece_player = array_idx(col, PLAYER_PAWNS)
                
                if idx_piece_player != -1 and idx_piece_player != len(col) - 1:
                    col[idx_piece_player] = CHECKPOINT_PLAYER
                elif idx_piece_with_checkpoint != -1 and idx_piece_with_checkpoint != len(col) - 1:
                    col[idx_piece_with_checkpoint] = CHECKPOINT_BOTH

                self.player_current_pawns = 3 - self.player_pawns_won
        else:
            for col in self.board:
                idx_piece_with_checkpoint = array_idx(col, CHECKPOINT_PAWNS_CURRENT_PLAYER)
                idx_piece_opponent = array_idx(col, OPPONENT_PAWNS)

                if idx_piece_opponent != -1 and idx_piece_opponent != len(col) - 1:
                    col[idx_piece_opponent] = CHECKPOINT_OPPONENT
                elif idx_piece_with_checkpoint != -1 and idx_piece_with_checkpoint != len(col) - 1:
                    col[idx_piece_with_checkpoint] = CHECKPOINT_BOTH

                self.opponent_current_pawns = 3 - self.opponent_pawns_won


    def print(self):
        for idx, col in enumerate(self.board):
            print(str(idx + 2), ":\t", col)


    def available_actions_mask(self) -> ndarray:
        mask = []

        for i, col in enumerate(self.board):
            mask.append([0 for _ in range(len(col))])
            
            idx_piece_player = array_idx(col, PLAYER_PAWNS)
            idx_piece_opponent = array_idx(col, OPPONENT_PAWNS)
            idx_checkpoint_player = array_idx(col, CHECKPOINT_PLAYER)
            idx_checkpoint_opponent = array_idx(col, CHECKPOINT_OPPONENT)
            idx_checkpoint = array_idx(col, CHECKPOINT_BOTH)
            idx_piece_with_checkpoint = array_idx(col, CHECKPOINT_PAWNS_CURRENT_PLAYER)

            if self.current_player == 0:    # Player turn
                nb_piece = self.player_current_pawns

                if idx_piece_opponent != (len(col) - 1) and idx_piece_player != (len(col) - 1):         # column not won by oponent or player
                    if nb_piece > 0:                                                                    # player has pieces to play
                        if idx_piece_player != -1:                                                      # player can continue his progress on this column
                            mask[i][idx_piece_player + 1] = 1

                        elif idx_checkpoint_player != -1:                                               # player have a checkpoint on this column
                            mask[i][idx_checkpoint_player] = 1

                        elif idx_checkpoint != -1:                                                      # player have a shared checkpoint on this column
                            mask[i][idx_checkpoint] = 1

                        elif idx_piece_with_checkpoint != -1:                                           # player have a piece with a opponent checkpoint on this column
                            mask[i][idx_piece_with_checkpoint + 1] = 1

                        elif idx_piece_player == -1 and idx_piece_opponent != 0:                        # player is not on the column or opponent didn't won the column
                            mask[i][0] = 1
                    else:

                        if idx_piece_player != -1 and idx_piece_player != len(col) - 1:                 # player can continue his progress on this column
                            mask[i][idx_piece_player + 1] = 1            

                        elif idx_piece_with_checkpoint != -1:                                           # player have a piece with a opponent checkpoint on this column
                            mask[i][idx_piece_with_checkpoint + 1] = 1
            else:                           # Opponent turn
                nb_piece = self.opponent_current_pawns

                if idx_piece_opponent != (len(col) - 1) and idx_piece_player != (len(col) - 1):         # column not won by oponent or player
                    if nb_piece > 0:                                                                    # opponent has pieces to play
                        if idx_piece_opponent != -1:                                                      # opponent can continue his progress on this column
                            mask[i][idx_piece_opponent + 1] = 1

                        elif idx_checkpoint_opponent != -1:                                               # opponent have a checkpoint on this column
                            mask[i][idx_checkpoint_opponent] = 1

                        elif idx_checkpoint != -1:                                                      # opponent have a shared checkpoint on this column
                            mask[i][idx_checkpoint] = 1

                        elif idx_piece_with_checkpoint != -1:                                           # opponent have a piece with a player checkpoint on this column
                            mask[i][idx_piece_with_checkpoint + 1] = 1

                        elif idx_piece_opponent == -1 and idx_piece_player != 0:                        # opponent is not on the column or player didn't won the column
                            mask[i][0] = 1
                    else:

                        if idx_piece_opponent != -1 and idx_piece_opponent != len(col) - 1:                 # opponent can continue his progress on this column
                            mask[i][idx_piece_opponent + 1] = 1            

                        elif idx_piece_with_checkpoint != -1:                                           # opponent have a piece with a player checkpoint on this column
                            mask[i][idx_piece_with_checkpoint + 1] = 1

        mask.append([1, 1])                                                                         # Continue turn or not, always available

        return mask
    
    def get_score(self) -> float:
        return self.score
    
    def get_game_over(self) -> bool:
        self.player_pawns_won = 0
        self.opponent_pawns_won = 0

        for col in self.board:
            if col[-1] == PLAYER_PAWNS:
                self.player_pawns_won += 1
            if col[-1] == OPPONENT_PAWNS:
                self.opponent_pawns_won += 1

        if self.player_pawns_won >= 3:
            self.game_over = True
            self.score = 1
         
        elif self.opponent_pawns_won >= 3: 
            self.game_over = True   
            self.score = -1 
    
    def get_obs(self) -> ndarray:
        return self.board
    
    def step(self, action: int):
        action = action - 2

        playable_index = self.get_playable_index(action)

        # Player turn
        if self.current_player == 0:
            if playable_index == -1:    # Move from the start
                self.board[action][0] = PLAYER_PAWNS
                self.player_current_pawns -= 1

            elif self.board[action][playable_index] == CHECKPOINT_PLAYER:
                if self.board[action][playable_index + 1] == CHECKPOINT_OPPONENT:
                    self.board[action][playable_index + 1] = CHECKPOINT_PAWNS_CURRENT_PLAYER
                else:
                    self.board[action][playable_index + 1] = PLAYER_PAWNS

                self.player_current_pawns -= 1
                self.board[action][playable_index] = 0

            elif self.board[action][playable_index] == PLAYER_PAWNS:
                if self.board[action][playable_index + 1] == CHECKPOINT_OPPONENT:
                    self.board[action][playable_index + 1] = CHECKPOINT_PAWNS_CURRENT_PLAYER
                else:
                    self.board[action][playable_index + 1] = PLAYER_PAWNS

                self.board[action][playable_index] = 0

            elif self.board[action][playable_index] == CHECKPOINT_PAWNS_CURRENT_PLAYER:
                if self.board[action][playable_index + 1] == CHECKPOINT_OPPONENT:
                    self.board[action][playable_index + 1] = CHECKPOINT_PAWNS_CURRENT_PLAYER
                else:
                    self.board[action][playable_index + 1] = PLAYER_PAWNS

                self.board[action][playable_index] = CHECKPOINT_OPPONENT

        # Opponent turn
        else:
            if playable_index == -1:    # Move from the start
                self.board[action][0] = OPPONENT_PAWNS
                self.opponent_current_pawns -= 1

            elif self.board[action][playable_index] == CHECKPOINT_PLAYER:
                if self.board[action][playable_index + 1] == CHECKPOINT_PLAYER:
                    self.board[action][playable_index + 1] = CHECKPOINT_PAWNS_CURRENT_PLAYER
                else:
                    self.board[action][playable_index + 1] = OPPONENT_PAWNS

                self.opponent_current_pawns -= 1
                self.board[action][playable_index] = 0

            elif self.board[action][playable_index] == OPPONENT_PAWNS:
                if self.board[action][playable_index + 1] == CHECKPOINT_PLAYER:
                    self.board[action][playable_index + 1] = CHECKPOINT_PAWNS_CURRENT_PLAYER
                else:
                    self.board[action][playable_index + 1] = OPPONENT_PAWNS

                self.board[action][playable_index] = 0

            elif self.board[action][playable_index] == CHECKPOINT_PAWNS_CURRENT_PLAYER:
                if self.board[action][playable_index + 1] == CHECKPOINT_PLAYER:
                    self.board[action][playable_index + 1] = CHECKPOINT_PAWNS_CURRENT_PLAYER
                else:
                    self.board[action][playable_index + 1] = OPPONENT_PAWNS

                self.board[action][playable_index] = CHECKPOINT_PLAYER
    

    def clone_stochastic():
        pass


    def play_turn(self):
        turn_over = False
        saved_board = self.board

        # Player turn
        if self.current_player == 0:

            while not turn_over:
                if not self.is_headless: print(f"Player turn: {self.player_current_pawns}")
                dices = self.roll_dices()

                actions_mask = self.available_actions_mask()
                
                self.print_mask(actions_mask)

                available_combinations = self.get_available_combinations(dices, actions_mask)

                if len(available_combinations) == 0:
                    turn_over = True
                    self.board = saved_board
                    break

                if self.player_is_random:
                    action = random.randint(0, len(available_combinations) - 1)
                    action = available_combinations[action]

                    self.step(action[0])

                    turn = ["y", "n"]

                    random_turn = random.randint(0, 1)

                    continue_turn = turn[random_turn]
                else:
                    # Input action

                    if not self.is_headless: print(f"Available combinations {str(dices)} ==> {str(available_combinations)}")

                    action = input("Choose the index of the action: ")

                    action = available_combinations[int(action)]

                    print("Chosen actions: ", action)

                    self.step(action[0])

                    continue_turn = input("Do you want to continue your turn? (y/n): ")

                if continue_turn == "n":
                    if not self.is_headless: print('End of turn\n')

                    self.save_checkpoint()

                    turn_over = True

                self.get_game_over()

                if self.game_over:
                    turn_over = True

                self.print() # Debug
                # print(f"\nPieces:  {str(self.player_current_pawns)} - Piece won {str(self.player_pawns_won)}") # Debug

        # Opponent turn   
        else:
            while not turn_over:
                if not self.is_headless: print(f"Opponent turn: {self.opponent_current_pawns}")
                dices = self.roll_dices()

                actions_mask = self.available_actions_mask()

                self.print_mask(actions_mask)

                available_combinations = self.get_available_combinations(dices, actions_mask)

                if len(available_combinations) == 0:
                    turn_over = True
                    self.board = saved_board
                    break

                if self.player_is_random:
                    action = random.randint(0, len(available_combinations) - 1)
                    action = available_combinations[action]

                    self.step(action[0])

                    turn = ["y", "n"]

                    random_turn = random.randint(0, 1)

                    continue_turn = turn[random_turn]
                else:
                    # Input action

                    if not self.is_headless: print(f"Available combinations {str(dices)} ==> {str(available_combinations)}")

                    action = input("Choose the index of the action: ")

                    action = available_combinations[int(action)]

                    print("Chosen actions: ", action)

                    self.step(action[0])

                    continue_turn = input("Do you want to continue your turn? (y/n): ")

                if continue_turn == "n":
                    if not self.is_headless: print('End of turn\n')

                    self.save_checkpoint()

                    turn_over = True

                self.get_game_over()

                if self.game_over:
                    turn_over = True

                self.print() # Debug
                # print(f"\nPieces:  {str(self.opponent_current_pawns)} - Piece won {str(self.opponent_pawns_won)}") # Debug


    def play(self):

        while not self.game_over:
            self.play_turn()
            print("Changement de joueur")
            self.current_player = 1 - self.current_player

        return self.score


game = CantStopEnv()
game.play()