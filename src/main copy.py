import numpy as np
import gym
import random
import os
from matplotlib import pyplot as plt
from agents.q_learning import QLearning, QLearningAgent
from envs.grid_world import GridWorldEnv
from envs.tic_tac_toe import TicTacToeEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from collections import deque


def create_q_model(input_shape, action_size):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))  # Action size = 9 pour Tic Tac Toe
    return model


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # facteur de remise
        self.epsilon = 1.0  # exploration initiale
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = create_q_model((state_size,), action_size)
        self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(available_actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # retourne l'action ayant la plus grande valeur Q
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    np.random.seed(42)

    env = TicTacToeEnv()
    agent = DQNAgent(env.OBS_SIZE, env.ACTION_SIZE)
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.OBS_SIZE])
        for time in range(500):
            available_actions = np.arange(env.ACTION_SIZE)[env.available_actions_mask().flatten() == 1]
            action = agent.act(state, available_actions)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.OBS_SIZE])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e+1}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > 32:
                agent.replay(32)


if __name__ == '__main__':
    main()

