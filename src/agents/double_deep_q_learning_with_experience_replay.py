import numpy as np
import time
import random
from src.plot_management import create_plots, create_score_plot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.losses import Huber
from src.agents.contracts import DeepAgent
from src.envs.contracts import DeepEnv
from collections import deque


def get_model(obs_size, action_size, learning_rate):
    model = Sequential()
    model.add(Dense(24, input_dim=obs_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model


def get_model_huber(obs_size, action_size, learning_rate):
    init = HeUniform()
    model = Sequential()
    model.add(Dense(24, input_shape=(obs_size,), activation='relu', kernel_initializer=init))
    model.add(Dense(12, activation='relu', kernel_initializer=init))
    model.add(Dense(action_size, activation='linear', kernel_initializer=init))
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model


class DoubleDeepQLearningAgentER(DeepAgent):
    def __init__(self, env: DeepEnv,
                 episodes: int = 100,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 max_steps: int = 1000,
                 max_memory: int = 2000,
                 batch_size: int = 64):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps = max_steps

        self.replay_memory = deque(maxlen=max_memory)
        self.batch_size = batch_size

        self.model = get_model_huber(self.env.OBS_SIZE, self.env.ACTION_SIZE, self.learning_rate)
        self.target_model = get_model_huber(self.env.OBS_SIZE, self.env.ACTION_SIZE, self.learning_rate)

        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, epsilon, action_mask):
        if np.random.rand() <= epsilon:
            return self.env.sample()

        state_reshaped = state.reshape([1, state.shape[0]])

        q_values = self.target_model.predict(state_reshaped) * action_mask
        min_q_value = np.min(q_values) - 1
        q_values_adjusted = q_values * action_mask + (1 - action_mask) * min_q_value

        return np.argmax(q_values_adjusted)

    def act(self) -> int:
        pass

    def train_model(self):

        if len(self.replay_memory) < self.batch_size:
            return

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + self.gamma * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q

            X.append(observation)
            Y.append(current_qs)

        self.model.fit(np.array(X), np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=True)

    def train(self):
        scores = []
        time_per_episode = []
        moving_average = []

        for e in range(self.episodes):
            start = time.time()
            state = self.env.reset()
            state = state.flatten()
            done = False
            score = 0
            step = 0

            while not done and step < self.max_steps:
                action_mask = self.env.available_actions_mask().flatten()

                action = self.choose_action(state, epsilon=self.epsilon, action_mask=action_mask)

                next_state, reward, done = self.env.step(action)
                next_state = next_state.flatten()

                self.replay_memory.append((state, action, reward, next_state, done))

                self.train_model()

                score += reward
                step += 1

                state = next_state.flatten()

            if step % 10 == 0:
                self.target_model.set_weights(self.model.get_weights())

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
        wins = 0
        looses = 0
        all_scores = []
        all_win_rates = []
        all_loss_rates = []
        episode_numbers = list(range(self.episodes))
        moving_average = []

        for e in range(self.episodes):
            state = self.env.reset()
            state = state.flatten()
            done = False
            score = 0
            step = 0

            while not done and step < self.max_steps:
                action_mask = self.env.available_actions_mask().flatten()

                action = self.choose_action(state, epsilon=self.epsilon, action_mask=action_mask)

                next_state, reward, done = self.env.step(action)
                next_state = next_state.flatten()

                score += reward
                step += 1

                state = next_state.flatten()

            print(f"Episode {e + 1}/{self.episodes}, score: {score}")
            all_scores.append(score)

            wins += 1 if self.env.get_game_result_status() == 1 else 0
            looses += 1 if self.env.get_game_result_status() == 0 else 0

            win_rate = wins / (e + 1)
            all_win_rates.append(win_rate)
            loss_rate = looses / (e + 1)
            all_loss_rates.append(loss_rate)

            moving_average.append(np.mean(all_scores[-100:]))

        average_score = np.mean(all_scores)
        print(f"Moyenne des scores sur {self.episodes} épisodes: {average_score}")
        print(f"Win rate: {all_win_rates[-1]}")
        print(f"Loss rate: {all_loss_rates[-1]}")

        create_plots(episode_numbers, all_win_rates, all_loss_rates)

        create_score_plot("Test", all_scores, moving_average)
