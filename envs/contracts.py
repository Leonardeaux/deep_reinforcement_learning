import numpy as np


class DeepEnv:
    def __init__(self, OBS_SIZE, ACTION_SIZE) -> None:
        pass

    def reset(self):
        pass

    def print(self):
        pass

    def available_actions_mask(self) -> np.ndarray:
        pass

    def get_score(self) -> float:
        pass

    def get_game_over(self) -> bool:
        pass

    def get_obs(self) -> np.ndarray:
        pass

    def step(self, action: int) -> tuple:
        pass

    def clone_stochastic():
        pass