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


def get_model_huber(obs_size, action_size, learning_rate):
    init = HeUniform()
    model = Sequential()
    model.add(Dense(24, input_shape=(obs_size,), activation='relu', kernel_initializer=init))
    model.add(Dense(12, activation='relu', kernel_initializer=init))
    model.add(Dense(action_size, activation='linear', kernel_initializer=init))
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model


class SumTree():
    """A summary tree data structure used in Prioritized Experience Replay."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, observation, action, reward, new_observation, done):
        """Add a transition to the sum-tree."""
        idx = self.capacity - 1
        self.data[idx] = (observation, action, reward, new_observation, done)
        self.update(idx, priority)

    def update(self, idx, priority):
        """Update the priority value at the given index."""
        parent = (idx - 1) // 2
        self.tree[parent] = self.tree[parent] + priority - self.tree[idx]
        if idx != 0:
            self.update(parent, priority)

    def get_leaf_node_idx(self, idx):
        """Get the leaf node index from the root or internal nodes."""
        return idx + self.capacity - 1

    def total_priority(self):
        """Return the sum of all priorities."""
        return self.tree[0]

    def sample(self, n, beta=0.4, epsilon=1e-6):
        """Sample transitions according to their priority weights."""
        p = self.total_priority() / n
        probs = p ** (-beta) / (p ** (-beta) + epsilon)
        samples = np.random.choice(range(n), size=n, replace=False, p=probs)
        observations, actions, rewards, new_observations, dones = [], [], [], [], []
        prios = np.empty(n, dtype=float)

        for i in samples:
            idx = self.get_leaf_node_idx(i)
            observation, action, reward, new_observation, done = self.data[idx]
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            new_observations.append(new_observation)
            dones.append(done)
            prio = (probs[i] + epsilon) ** (-1 / beta)
            prios[i] = prio
            self.update(idx, prio)

        return np.stack(observations), np.stack(actions), np.stack(rewards).astype(np.float32), \
            np.stack(new_observations), np.stack(dones), prios


class BetaDistributionSampler():
    """Beta distribution sampler used to adjust the probability of selecting experiences."""

    def __init__(self, alpha=0.6, beta=0.4):
        self.alpha = alpha
        self.beta = beta

    @property
    def sample(self):
        """Sample a number between 0 and 1 using the specified parameters."""
        return np.random.beta(self.alpha, self.beta)


class DoubleDeepQLearningAgentPrioritizedER(DeepAgent):
    # ... previous methods unchanged here ...

    def __init__(self, env: DeepEnv,
                 episodes: int = 100,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 max_steps: int = 1000,
                 max_memory: int = 2000,
                 batch_size: int = 64,
                 alpha: float = 0.6,
                 beta: float = 0.4):

        super().__init__(env)
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps = max_steps

        self.replay_memory = PriorityReplayBuffer(max_memory)
        self.batch_size = batch_size
        self.sampler = BetaDistributionSampler(alpha, beta)

        self.model = get_model_huber(self.env.OBS_SIZE, self.env.ACTION_SIZE, self.learning_rate)
        self.target_model = get_model_huber(self.env.OBS_SIZE, self.env.ACTION_SIZE, self.learning_rate)

        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state, epsilon, action_mask):
        pass

    def act(self) -> int:
        pass

    def _add_to_memory(self, transition, priority):
        """Add a transition to the memory buffer with its associated priority."""
        self.replay_memory.push(transition, priority)

    def train_model(self):
        """Unchanged method"""

    def train(self):
        """Modified training loop to include priority updates."""
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

                priority = self._calculate_priority(state, action, reward, next_state, done)
                self._add_to_memory((state, action, reward, next_state, done), priority)

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

    def _calculate_priority(self, state, action, reward, next_state, done):
        """Calculate the priority value for the given transition."""
        _, _, max_Q_next_state = self.target_model.predict([next_state])
        target_Q = reward + (not done) * self.gamma * np.amax(max_Q_next_state)
        current_Q = self.model.predict([state])[:, action]
        error = abs(target_Q - current_Q)
        return (error + 1) ** 0.5


class PriorityReplayBuffer():
    """Memory buffer implementing Prioritized Experience Replay."""

    def __init__(self, capacity):
        self.sum_tree = SumTree(capacity)

    def push(self, transition, priority):
        """Store a new transition into the buffer along with its priority."""
        self.sum_tree.add(priority, *transition)

    def sample(self, batch_size):
        """Retrieve a set of transitions randomly selected based on their priority."""
        obs, acts, rews, new_obs, dones, prios = self.sum_tree.sample(batch_size)
        return obs, acts, rews, new_obs, dones, prios

    def clear(self):
        """Clear out all stored transitions."""
        self.sum_tree = SumTree(self.sum_tree.capacity)