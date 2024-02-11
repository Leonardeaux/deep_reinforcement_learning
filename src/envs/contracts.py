import numpy as np


class DeepEnv:
    def __init__(self, OBS_SIZE, ACTION_SIZE) -> None:
        self.OBS_SIZE = OBS_SIZE
        self.ACTION_SIZE = ACTION_SIZE
        self.nb_player = None

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

    def step(self, action: int):
        pass

    def step_play(self, action: int):
        pass

    def clone_stochastic(self):
        pass

    def available_actions(self):
        pass

    def sample(self):
        pass

    def get_game_result_status(self):
        pass
