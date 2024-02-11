import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from tqdm import tqdm
from src.agents.contracts import DeepAgent
from src.envs.contracts import DeepEnv


class RandomAgent(DeepAgent):
    def __init__(self, env: DeepEnv, episodes: int = 100):
        super().__init__(env)
        self.episodes = episodes

    def act(self):
        pass

    def train(self):
        pass

    def test(self):
        scores = []
        wins = 0
        looses = 0
        draws = 0

        for e in range(self.episodes):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                action = self.env.sample()

                _, reward, done = self.env.step(action)

                score += reward

            print(f"Episode {e + 1}/{self.episodes}, score: {score}")
            scores.append(score)
            wins += 1 if self.env.get_game_result_status() == 1 else 0
            looses += 1 if self.env.get_game_result_status() == 0 else 0
            draws += 1 if self.env.get_game_result_status() == 0.5 else 0

        print(f"Winrate: {wins/self.episodes}")
        print(f"Loose rate: {looses/self.episodes}")
        print(f"Draw rate: {draws/self.episodes}")
        average_score = np.mean(scores)
        print(f"Moyenne des scores sur {self.episodes} épisodes: {average_score}")
