import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from src.plot_management import create_plots
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
        wins = 0
        looses = 0
        all_scores = []
        all_win_rates = []
        all_loss_rates = []
        episode_numbers = list(range(self.episodes))

        for e in range(self.episodes):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                action = self.env.sample()

                _, reward, done = self.env.step(action)

                score += reward

            print(f"Episode {e + 1}/{self.episodes}, score: {score}")
            all_scores.append(score)

            wins += 1 if self.env.get_game_result_status() == 1 else 0
            looses += 1 if self.env.get_game_result_status() == 0 else 0

            win_rate = wins / (e + 1)
            all_win_rates.append(win_rate)
            loss_rate = looses / (e + 1)
            all_loss_rates.append(loss_rate)

        average_score = np.mean(all_scores)
        print(f"Moyenne des scores sur {self.episodes} épisodes: {average_score}")

        create_plots(episode_numbers, all_scores, all_win_rates, all_loss_rates)
