import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
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


class DoubleDeepQLearningExpRepAgent(DeepAgent):
    def __init__(self, env: DeepEnv,
                 episodes: int = 100,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 max_steps: int = 1000):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.gamma = gamma
        self.max_steps = max_steps

        self.model = get_model(self.env.OBS_SIZE, self.env.ACTION_SIZE, self.learning_rate)

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
        target = reward

        if not done:
            next_q_values = self.model.predict(next_state)

            target = reward + self.gamma * np.amax(next_q_values)
        target_f = self.model.predict(state)
        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)

    def train(self):
        scores = []

        for e in range(self.episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.env.OBS_SIZE])
            done = False
            score = 0
            step = 0

            while not done and step < self.max_steps:
                action_mask = np.reshape(self.env.available_actions_mask(), [1, self.env.ACTION_SIZE])

                action = self.choose_action(state, epsilon=0.2, action_mask=action_mask)

                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.OBS_SIZE])

                score += reward
                self.train_model(state, action, reward, next_state, done)

                state = np.reshape(next_state, [1, self.env.OBS_SIZE])

                step += 1

            print(f"Episode {e + 1}/{self.episodes}, score: {score}")
            scores.append(score)

        average_score = np.mean(scores)
        print(f"Moyenne des scores sur {self.episodes} épisodes: {average_score}")

    def test(self):
        pass
