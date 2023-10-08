from envs.contracts import DeepEnv
import pygame
import numpy as np


class LineWorld(DeepEnv):
    def __init__(self, OBS_SIZE, ACTION_SIZE, win_coord: (int, int), lose_coord: (int, int)) -> None:
        self.length = OBS_SIZE
        self.state = None
        self.goal = win_coord
        self.reset()

    def reset(self):
        self.state = np.random.randint(self.length)
        self.goal = np.random.choice([i for i in range(self.length) if i != self.state])
        return self.state
    
    def print(self):
        pass

    def available_actions_mask(self) -> np.ndarray:
        pass


    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(self.length - 1, self.state + 1)

        done = self.state == self.goal
        reward = 1 if done else 0
        return self.state, reward, done

    def get_state(self):
        return self.state
    

if __name__ == "__main__":
    pass
