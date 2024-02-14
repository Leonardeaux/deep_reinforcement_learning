import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import random
import tqdm
import os
import tensorflow as tf
from envs.grid_world import GridWorldEnv
from envs.tic_tac_toe import TicTacToeEnv
from agents.mcts import MCTSAgent
from agents.random_rollout import RandomRolloutAgent
from agents.mcts_gym import MCTSAgent as MCTSAgentGym
from agents.q_learning import QLearningAgent
from agents.deep_q_learning import DeepQLearningAgent
from agents.double_deep_q_learning import DoubleDeepQLearningAgent
from agents.double_deep_q_learning_with_experience_replay import DoubleDeepQLearningAgentER
from agents.double_deep_q_learning_with_prioritized_experience_replay import DoubleDeepQLearningAgentPrioritizedER
from agents.random import RandomAgent
from collections import deque


def main():
    tf.config.set_visible_devices([], 'GPU')
    np.random.seed(42)

    env = TicTacToeEnv()
    env3 = GridWorldEnv(size=5)
    episodes = 5

    # agent = QLearningAgent(env, episodes=episodes, alpha=0.1, gamma=0.99, epsilon=0.1)
    # agent = DeepQLearningAgent(env, episodes=episodes, learning_rate=0.001)
    # agent = DoubleDeepQLearningAgent(env, episodes=episodes, learning_rate=0.01)
    # agent = DoubleDeepQLearningAgentER(env, episodes=episodes, learning_rate=0.001)
    agent = DoubleDeepQLearningAgentPrioritizedER(env, episodes=episodes, learning_rate=0.001)

    scores, time_per_episode = agent.train()
    agent.test()
    #
    # print()
    # print("---------------------------------------------------------------------------------")
    # average_score = np.mean(scores)
    # print("Train Metrics")
    # print(f"Moyenne des scores sur {episodes} épisodes: {average_score}")
    # average_time = np.mean(time_per_episode)
    # print(f"Temps moyen d'entraînement par épisode : {round(average_time, 2)} secondes.")


if __name__ == '__main__':
    main()
