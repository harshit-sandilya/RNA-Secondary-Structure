import gymnasium as gym
from gymnasium import spaces
import numpy as np


def get_energy(a, b):
    if a == 2 and b == 3 or a == 3 and b == 2:
        return 3
    if a == 1 and b == 4 or a == 4 and b == 1:
        return 2
    if a == 3 and b == 4 or a == 4 and b == 3:
        return 2
    return 0


class Environment(gym.Env):
    def __init__(self, sample, output):
        super(Environment, self).__init__()
        self.sample = sample
        self.output = output
        self.action_space = spaces.Box(low=0, high=3, shape=(31,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(31,), dtype=np.float32
        )
        self.batch_size = sample.shape[0]
        self.time_steps = sample.shape[1]
        self.current_step = 0
        self.current_batch = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_batch = 0
        return self.sample[self.current_batch, self.current_step], {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.time_steps
        if not done:
            observation = self.sample[self.current_batch, self.current_step]
        else:
            self.current_step = 0
            self.current_batch += 1
            if self.current_batch >= self.batch_size:
                observation = np.zeros_like(self.sample[0, 0])
                done = True
            else:
                observation = self.sample[self.current_batch, self.current_step]
        reward = self.calculate_reward(action)
        return observation, reward, done, False, {}

    # def calculate_reward(self, structure, sequence):
    #     structure = np.round(structure)
    #     reward = 0
    #     left = 0
    #     right = len(structure) - 1
    #     while left < right:
    #         if (
    #             structure[left] == 2
    #             and structure[right] == 3
    #             or structure[left] == 3
    #             and structure[right] == 2
    #         ):
    #             reward += get_energy(sequence[left], sequence[right])
    #         left += 1
    #         right -= 1
    #     return reward

    def calculate_reward(self, action):
        target_value = self.output[self.current_batch, self.current_step]
        predicted_value = action.round()
        accuracy = np.mean(np.equal(predicted_value, target_value))
        reward = accuracy * 100
        return reward
