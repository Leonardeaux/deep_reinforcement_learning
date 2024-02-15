import numpy as np
import random
from src.envs.contracts import DeepEnv
from src.agents.random import RandomAgent


class BalloonPopEnv(DeepEnv):
    def __init__(self):
        # RED = 0, YELLOW = 1, BLUE = 2
        # MOON = 3, STAR = 4, DIAMOND = 5

        OBS_SIZE = 11
        ACTION_SIZE = 32
        super().__init__(OBS_SIZE, ACTION_SIZE)
        self.colors = [0, 1, 2]  # ['YELLOW', 'BLUE', 'RED']
        self.symbols = [30, 40, 50]  # ['STAR', 'MOON', 'DIAMOND']
        self.dice_nb = 3
        self.dice_tuple = [
            (2, 30),
            (2, 40),
            (2, 50),
            (1, 30),
            (1, 40),
            (0, 30)
        ]
        self.score_sheet = {
            0: [0, 3, 7, 11, 15, 3],
            1: [1, 3, 5, 7, 9, 12, 8],
            2: [0, 0, 0, 2, 4, 6, 8, 10, 14, 6],
            30: [1, 2, 3, 5, 7, 10, 13, 16, 4],
            40: [2, 3, 4, 5, 7, 9, 12, 5],
            50: [1, 3, 6, 10, 13, 7]
        }
        self.current_scores = None
        self.final_scores = None
        self.game_over = None
        self.dice = None
        self.reset()

    def reset(self):
        self.current_scores = {key: -1 for key in self.score_sheet}
        # self.current_scores = {0: 5, 1: 2, 2: 9, 30: 8, 40: 7, 50: 5}
        self.final_scores = {}
        self.game_over = False
        self.dice = []

        self.roll_dice()

        return self.get_obs()

    def print(self):
        print("\nCurrent score sheet:")
        for key, values in self.score_sheet.items():
            current_position = self.current_scores[key]
            score_line = ' '.join(f"{value:2d}" for value in values)
            progress_line = '*' * current_position + '.' * (len(values) - current_position)
            print(f"{key}: {progress_line} ({current_position}/{len(values)}) - Scores: {score_line}")

    def get_obs(self):
        obs = self.get_state() + self.get_dice_rolls()

        add = self.OBS_SIZE - len(obs)

        return np.array(obs + [0] * add)

    def get_dice_rolls(self):
        dice = []
        for d in self.dice:
            dice.append(d[0] + d[1])

        return dice

    def get_state(self):
        state = []
        for key in self.current_scores:
            state.append(self.current_scores[key])

        return state

    def roll_dice(self):
        for i in range(self.dice_nb):
            # random of a combination of color and symbol
            self.dice.append(random.choice(self.dice_tuple))

    def available_actions(self) -> np.ndarray:
        return np.arange(self.ACTION_SIZE)[self.available_actions_mask() == 1]

    def available_actions_mask(self) -> np.ndarray:
        mask = np.zeros(self.ACTION_SIZE)

        if self.dice_nb == 5:
            mask[0] = 1

            return mask

        # 2^5 combinations of reroll decisions
        combinations_nb = 2 ** self.dice_nb
        for i in range(combinations_nb):
            mask[i] = 1

        return mask

    def sample(self):
        available_actions = np.arange(self.ACTION_SIZE)[self.available_actions_mask() == 1]
        return np.random.choice(available_actions)

    def step(self, action):
        # Decode action into dice to keep and reroll
        dice_to_reroll = []

        for i in range(self.dice_nb):
            if action & (1 << i):
                dice_to_reroll.append(self.dice[i])
                self.dice[i] = random.choice(self.dice_tuple)

        if len(dice_to_reroll) > 0:
            self.dice.append(random.choice(self.dice_tuple))
            self.dice_nb += 1
        else:
            for i in range(self.dice_nb):
                # verify if the dice get me in a max position of the column in the score sheet
                if self.current_scores[self.dice[i][0]] < len(self.score_sheet[self.dice[i][0]]) - 1:
                    self.current_scores[self.dice[i][0]] += 1

                if self.current_scores[self.dice[i][1]] < len(self.score_sheet[self.dice[i][1]]) - 1:
                    self.current_scores[self.dice[i][1]] += 1

        # Check for game over
        # If they are 3 columns filled, the game is over
        for key, values in self.score_sheet.items():
            if len(self.final_scores) == 3:
                self.game_over = True
                break

            if self.current_scores[key] == len(values) - 1 and key not in self.final_scores.keys():
                self.final_scores[key] = sum(values)

        return self.get_obs(), self.calculate_reward(), self.get_game_over()

    def step_play(self, action):
        return self.step(action)

    def get_game_over(self) -> bool:
        return self.game_over

    def calculate_reward(self):
        if self.game_over:

            rewards_sum = sum(self.final_scores.values())

            return rewards_sum

        return 0

    def get_game_result_status(self):
        if self.get_game_over():
            if self.get_score() > 160:
                return 1
            else:
                return 0
        else:
            return 0


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    env = BalloonPopEnv()
    print(env.reset())
    print(env.get_obs())
    mask = env.available_actions_mask()
    print(mask)
    i = 0
    while not env.game_over:
        i += 1
        print(f"Step {i}")
        action = 0
        env.print()
        print(env.step(action))
        env.print()
        print('available_actions_mask: ', env.available_actions_mask())
        print('available_actions: ', env.available_actions())
        print('sample: ', env.sample())
        print('final_scores: ', env.final_scores)
