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
from envs.balloon_pop import BalloonPopEnv
from agents.mcts import MCTSAgent
from agents.random_rollout import RandomRolloutAgent
from agents.q_learning import QLearningAgent
from agents.deep_q_learning import DeepQLearningAgent
from agents.double_deep_q_learning import DoubleDeepQLearningAgent
from agents.double_deep_q_learning_with_experience_replay import DoubleDeepQLearningAgentER
from agents.random import RandomAgent
from collections import deque


def main():
    tf.config.set_visible_devices([], 'GPU')
    np.random.seed(42)

    env = TicTacToeEnv()

    episodes = 500
    agent = MCTSAgent(env, episodes=episodes, max_depth=200)

    scores, time_per_episode = agent.train()
    agent.test()

    print()
    print("---------------------------------------------------------------------------------")
    average_score = np.mean(scores)
    print("Train Metrics")
    print(f"Moyenne des scores sur {episodes} épisodes: {average_score}")
    average_time = np.mean(time_per_episode)
    print(f"Temps moyen d'entraînement par épisode : {round(average_time, 2)} secondes.")


if __name__ == '__main__':
    main()
