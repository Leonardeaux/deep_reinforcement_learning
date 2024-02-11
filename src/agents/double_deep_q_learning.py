import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.losses import Huber
from src.agents.contracts import DeepAgent
from src.envs.contracts import DeepEnv


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
    model.add(Dense(24, input_shape=obs_size, activation='relu', kernel_initializer=init))
    model.add(Dense(12, activation='relu', kernel_initializer=init))
    model.add(Dense(action_size, activation='linear', kernel_initializer=init))
    model.compile(loss=Huber(), optimizer=Adam(lr=learning_rate),
                  metrics=['accuracy'])

    return model


class DoubleDeepQLearningAgent(DeepAgent):
    def __init__(self, env: DeepEnv,
                 episodes: int = 100,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 max_steps: int = 1000):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps = max_steps

        self.model = get_model(self.env.OBS_SIZE, self.env.ACTION_SIZE, self.learning_rate)
        self.target_model = get_model(self.env.OBS_SIZE, self.env.ACTION_SIZE, self.learning_rate)

        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, epsilon, action_mask):
        if np.random.rand() <= epsilon:
            return self.env.sample()

        q_values = self.model.predict(state) * action_mask
        min_q_value = np.min(q_values) - 1
        q_values_adjusted = q_values * action_mask + (1 - action_mask) * min_q_value

        return np.argmax(q_values_adjusted)

    def act(self) -> int:
        pass

    def train_model(self, state, action, reward, next_state, done):

        current_qs = self.model.predict(state)

        future_qs = self.target_model.predict(next_state)

        if not done:
            max_future_q = reward + self.gamma * np.max(future_qs)
        else:
            max_future_q = reward

        current_qs[0][action] = (1 - self.learning_rate) * current_qs[0][action] + self.learning_rate * max_future_q

        self.model.fit(state, current_qs, verbose=0)

    def train(self):
        scores = []
        time_per_episode = []

        for e in range(self.episodes):
            start = time.time()
            state = self.env.reset()
            state = np.reshape(state, [1, self.env.OBS_SIZE])
            done = False
            score = 0
            step = 0

            while not done and step < self.max_steps:
                action_mask = np.reshape(self.env.available_actions_mask(), [1, self.env.ACTION_SIZE])

                action = self.choose_action(state, epsilon=self.epsilon, action_mask=action_mask)

                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.OBS_SIZE])

                self.train_model(state, action, reward, next_state, done)

                score += reward
                step += 1

                state = np.reshape(next_state, [1, self.env.OBS_SIZE])

            if step % 10 == 0:
                self.target_model.set_weights(self.model.get_weights())

            print(f"Episode {e + 1}/{self.episodes}, score: {score}")
            scores.append(score)

            end = time.time()
            time_per_episode.append(end - start)

        average_score = np.mean(scores)
        print(f"Moyenne des scores sur {self.episodes} épisodes: {average_score}")
        average_time = np.mean(time_per_episode)
        print(f"Temps moyen d'entraînement par épisode : {round(average_time, 2)} secondes.")

        return scores, time_per_episode

    def test(self):
        pass
