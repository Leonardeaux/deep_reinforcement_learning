import random
import numpy as np
from matplotlib import pyplot as plt
from math import log, sqrt
from typing import Tuple, Any
from copy import deepcopy
from src.envs.contracts import DeepEnv
from src.agents.contracts import DeepAgent


class Node:
    def __init__(self, env: DeepEnv, done: bool, parent: Any, obs, action_index: int, UCBc: float):
        self.actions_size = env.ACTION_SIZE
        self.obs_size = env.OBS_SIZE
        # self.actions_size = env.action_space.n
        # self.obs_size = env.observation_space
        self.child = None
        self.total_rewards = 0
        self.visits = 0
        self.env = env
        self.obs = obs
        self.done = done
        self.parent = parent
        self.action_index = action_index
        self.UCBc = UCBc

    def get_uc_bscore(self) -> float:
        if self.visits == 0:
            return float('inf')

        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        return (self.total_rewards / self.visits) + self.UCBc * sqrt(log(top_node.visits) / self.visits)

    def remove_parent(self):
        del self.parent
        self.parent = None

    def create_children(self):
        if self.done:
            return

        actions = []
        games = []

        actions_available = self.env.available_actions()

        # for each action available, create a new game and add it to the list of games
        for i in range(self.actions_size):
            if i in actions_available:
                actions.append(i)
                new_game = deepcopy(self.env)
                games.append(new_game)

        child = {}
        for action, game in zip(actions, games):
            observation, reward, done = game.step_play(action)
            child[action] = Node(game, done, self, observation, action, self.UCBc)

        self.child = child

    def explore(self):
        current_node = self

        # Selection des nodes niveau fin
        while current_node.child:
            child = current_node.child

            max_U = max(ch.get_uc_bscore() for ch in child.values())
            actions = [a for a, ch in child.items() if ch.get_uc_bscore() == max_U]

            if len(actions) == 0:
                print("Error no max score action ", max_U)

            # Si plusieurs action avec le max score prendre au hasard
            action = random.choice(actions)
            current_node = child[action]

        if current_node.visits < 1:
            current_node.total_rewards = current_node.total_rewards + current_node.rollout()
        else:
            current_node.create_children()
            if current_node.child:
                # current = a random child
                # random.choice(current.child) is not working
                current_node = random.choice(list(current_node.child.values()))
            current_node.total_rewards = current_node.total_rewards + current_node.rollout()

        current_node.visits += 1

        parent = current_node

        while parent.parent:
            parent = parent.parent
            parent.visits += 1
            parent.total_rewards = parent.total_rewards + current_node.total_rewards

    def rollout(self) -> int:
        if self.done:
            return 0

        v = 0
        done = False
        new_game = deepcopy(self.env)

        while not done:
            # Action disponible random
            action = new_game.sample()
            observation, reward, done = new_game.step_play(action)
            v = v + reward

        new_game.reset()

        return v

    def next(self) -> Tuple['Node', int]:
        if self.done:
            raise ValueError("Game has ended")

        if not self.child:
            raise ValueError('No children found and game hasn\'t ended')

        child = self.child

        available_action = self.env.available_actions()

        for a, ch in child.items():
            if a not in available_action:
                del child[a]

        max_visits = max(node.visits for node in child.values())

        max_children = [ch for a, ch in child.items() if ch.visits == max_visits]

        if len(max_children) == 0:
            print("Error no max visits action ", max_visits)

        # si plusieurs action avec le max de visits prendre au hasard
        max_child = random.choice(max_children)

        return max_child, max_child.action_index


class MCTSAgent(DeepAgent):
    def __init__(self, env: DeepEnv, episodes: int = 10, max_depth: int = 100, UCBc: float = 1.0):
        self.principal_env = env
        self.episodes = episodes
        self.max_depth = max_depth
        self.UCBc = UCBc

    def act(self):
        pass

    def train(self):
        rewards = []
        moving_average = []

        for e in range(self.episodes):
            reward_e = 0
            principal_env = self.principal_env
            obs = principal_env.reset()
            done = False

            new_game = deepcopy(principal_env)
            mytree = Node(new_game, False, 0, obs, 0, self.UCBc)

            print('Episode: ', e + 1)

            while not done:
                for _ in range(self.max_depth):
                    mytree.explore()

                mytree, action = mytree.next()

                mytree.remove_parent()

                obs, reward, done = principal_env.step_play(action)

                if done:
                    reward_e = reward_e + reward
                    break

                if principal_env.nb_player > 1:
                    obs, reward, done = principal_env.step_play(principal_env.sample())

                    new_game = deepcopy(principal_env)
                    mytree = Node(new_game, new_game.get_game_over(), mytree, obs, mytree.action_index, self.UCBc)

                reward_e = reward_e + reward

            print("Reward de l'épisode: ", reward_e)

            rewards.append(reward_e)
            moving_average.append(np.mean(rewards[-100:]))

        plt.plot(rewards)
        plt.plot(moving_average)
        plt.show()
        print('Reward total moyen: ' + str(np.mean(rewards)))

    def test(self, num_simulation: int = 1000):
        pass
