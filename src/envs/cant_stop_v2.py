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
CHECKPOINT_PAWNS_PLAYER = 6     # Player have a pawn on a checkpoint
CHECKPOINT_PAWNS_OPPONENT = 7   # Opponent have a pawn on a checkpoint


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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0]
        ]

        self.player_pawns_won = 0
        self.opponent_pawns_won = 0

        self.player_current_pawns = 3
        self.opponent_current_pawns = 3

    def print(self):
        for idx, col in enumerate(self.board):
            print(str(idx+2), ":\t", col)

    def available_actions_mask(self) -> ndarray:
        mask = []

        for col in self.board:
            mask.append([0 for _ in range(len(self.board[col]))])
            
            idx_piece_player = array_idx(self.board[col], PLAYER_PAWNS)
            idx_piece_opponent = array_idx(self.board[col], OPPONENT_PAWNS)

            idx_checkpoint_player = array_idx(self.board[col], CHECKPOINT_PLAYER)
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
        pass
    
    def get_game_over(self) -> bool:
        pass
    
    def get_obs(self) -> ndarray:
        pass
    
    def step(self, action: int):
        pass
    
    def clone_stochastic():
        pass


game = CantStopEnv()
game.print()