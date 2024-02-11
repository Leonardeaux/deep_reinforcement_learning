import random
import numpy as np
from math import log, sqrt
from typing import Tuple
from copy import deepcopy
from envs.contracts import DeepEnv
from agents.contracts import DeepAgent


class Node:
    def __init__(self, env: DeepEnv, done: bool, parent, obs, action_index: int, UCBc):
        self.actions_size = env.action_space.n
        self.obs_size = env.observation_space
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

    def create_child(self):
        if self.done:
            return

        actions = []
        games = []

        for i in range(self.actions_size):
            actions.append(i)
            new_game = deepcopy(self.env)
            games.append(new_game)

        child = {}
        for action, game in zip(actions, games):
            observation, reward, done, _, _ = game.step(action)
            child[action] = Node(game, done, self, observation, action, self.UCBc)

        self.child = child

    def explore(self):
        current = self

        while current.child:

            child = current.child
            max_U = max(ch.get_uc_bscore() for ch in child.values())
            actions = [a for a, ch in child.items() if ch.get_uc_bscore() == max_U]

            if len(actions) == 0:
                print("Error no action available ", max_U)

            action = random.choice(actions)
            current = child[action]

        if current.visits < 1:
            current.total_rewards = current.total_rewards + current.rollout()
        else:
            current.create_child()
            if current.child:
                current = random.choice(current.child)
            current.total_rewards = current.total_rewards + current.rollout()

        current.visits += 1

        parent = current

        while parent.parent:
            parent = parent.parent
            parent.visits += 1
            parent.total_rewards = parent.total_rewards + current.total_rewards

    def rollout(self) -> int:
        if self.done:
            return 0

        v = 0
        done = False
        new_game = deepcopy(self.env)

        while not done:
            action = new_game.action_space.sample()
            observation, reward, done, _, _ = new_game.step(action)
            v = v + reward

        new_game.reset()

        return v

    def next(self) -> Tuple['Node', int]:
        if self.done:
            raise ValueError("Game has ended")

        if not self.child:
            raise ValueError('No children found and game hasn\'t ended')

        child = self.child

        max_visits = max(node.visits for node in child.values())

        max_children = [ch for a, ch in child.items() if ch.visits == max_visits]

        if len(max_children) == 0:
            print("error zero length ", max_visits)

        max_child = random.choice(max_children)

        return max_child, max_child.action_index


class MCTSAgent(DeepAgent):
    def __init__(self, env: DeepEnv, episodes: 10, max_depth: int = 100, UCBc=1.0):
        self.env = env
        self.episodes = episodes
        self.max_depth = max_depth
        self.UCBc = UCBc

    def act(self, state: np.ndarray) -> int:
        pass

    def train(self):
        rewards = []
        moving_average = []

        for e in range(self.episodes):

            reward_e = 0
            env = self.env
            obs = env.reset()
            done = False

            new_game = deepcopy(env)
            mytree = Node(new_game, False, 0, obs, 0, self.UCBc)

            print('episode #' + str(e + 1))

            while not done:

                for _ in range(self.max_depth):
                    mytree.explore()

                next_tree, next_action = mytree.next()

                next_tree.remove_parent()

                mytree, action = next_tree, next_action

                obs, reward, done, _, _ = env.step(action)

                reward_e = reward_e + reward

                if done:
                    print('reward_e ' + str(reward_e))
                    break

            rewards.append(reward_e)
            moving_average.append(np.mean(rewards[-100:]))

    def test(self, num_simulation: int = 1000):
        pass
