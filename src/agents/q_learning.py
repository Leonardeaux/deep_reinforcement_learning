import numpy as np
import time
from src.plot_management import create_plots, create_score_plot
from tqdm import tqdm
from src.agents.contracts import DeepAgent
from src.envs.contracts import DeepEnv
import matplotlib.pyplot as plt


class QLearningAgent(DeepAgent):
    def __init__(self, env: DeepEnv, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=1000, max_steps=1000):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps = max_steps
        self.Q = {}

    def state_to_key(self, state):
        if isinstance(state, tuple):
            return str(state)
        elif isinstance(state, int):
            return str(state)
        elif isinstance(state, list):
            return str(state)
        else:
            return str(state.flatten().tolist())

    def get_Q(self, state, action):
        key = self.state_to_key(state)

        if key not in self.Q:
            self.Q[key] = np.zeros(self.env.ACTION_SIZE)

        return self.Q[key][action]

    def set_Q(self, state, action, value):
        key = self.state_to_key(state)

        if key not in self.Q:
            self.Q[key] = np.zeros(self.env.ACTION_SIZE)

        self.Q[key][action] = value

    def choose_action(self, state):
        mask = self.env.available_actions_mask().flatten()
        available_actions = np.arange(self.env.ACTION_SIZE)[mask == 1]

        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            Q_values = np.array([self.get_Q(state, a) for a in available_actions])

            return available_actions[np.argmax(Q_values)]

    def update(self, state, action, reward, next_state, done):
        current_Q = self.get_Q(state, action)

        if done:
            target_Q = reward
        else:
            next_Q_values = [self.get_Q(next_state, a)
                             for a in np.arange(self.env.ACTION_SIZE)
                             [self.env.available_actions_mask().flatten() == 1]]
            target_Q = reward + self.gamma * np.max(next_Q_values)

        updated_Q = current_Q + self.alpha * (target_Q - current_Q)
        self.set_Q(state, action, updated_Q)

    def act(self):
        pass

    def train(self):
        scores = []
        time_per_episode = []
        moving_average = []

        for e in tqdm(range(self.episodes)):
            start = time.time()
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done and step < self.max_steps:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                step += 1
                score += reward

            print(f"Episode {e + 1}/{self.episodes}, score: {score}")
            scores.append(score)

            end = time.time()
            time_per_episode.append(end - start)

            moving_average.append(np.mean(scores[-100:]))

        average_score = np.mean(scores)
        print(f"Moyenne des scores sur {self.episodes} épisodes: {average_score}")

        average_time = np.mean(time_per_episode)
        print(f"Temps moyen d'entraînement par épisode : {round(average_time, 2)} secondes.")

        create_score_plot("Train", scores, moving_average)

        return scores, time_per_episode

    def test(self):
        episodes = 500
        wins = 0
        looses = 0
        all_scores = []
        all_win_rates = []
        all_loss_rates = []
        episode_numbers = list(range(episodes))
        moving_average = []

        for e in tqdm(range(episodes)):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done and step < self.max_steps:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                state = next_state
                step += 1
                score += reward

            print(f"Episode {e + 1}/{episodes}, score: {score}")
            all_scores.append(score)

            wins += 1 if self.env.get_game_result_status() == 1 else 0
            looses += 1 if self.env.get_game_result_status() == 0 else 0

            win_rate = wins / (e + 1)
            all_win_rates.append(win_rate)
            loss_rate = looses / (e + 1)
            all_loss_rates.append(loss_rate)

            moving_average.append(np.mean(all_scores[-100:]))

        average_score = np.mean(all_scores)
        print(f"Moyenne des scores sur {episodes} épisodes: {average_score}")
        print(f"Win rate: {all_win_rates[-1]}")
        print(f"Loss rate: {all_loss_rates[-1]}")

        create_plots(episode_numbers, all_win_rates, all_loss_rates)

        create_score_plot("Test", all_scores, moving_average)
