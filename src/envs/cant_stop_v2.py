import numpy as np
import random
from contracts import DeepEnv
from Player import Player

class CantStopEnv(DeepEnv):
    def __init__(self, is_headless: bool = False):
        self.is_headless = is_headless
        self.OBS_SIZE = 83  # Nombre de case
        self.ACTION_SIZE = 83*2  # Nombre de case * 2 (continuer ou non)
        self.player = Player("RED", False, "player")
        self.opponent = Player("GREEN", False, "opponent")
        self.reset()

    def reset(self):
        self.board = np.array([
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
        ])


game = CantStopEnv()