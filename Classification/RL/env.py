import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Environment(gym.Env):
    def __init__(self, sample, output):
        super(Environment, self).__init__()
        self.sample = sample
        self.output = output
        self.action_space = spaces.Discrete(4, start=0)
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
        reward = self.calculate_reward(action)
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
        return observation, reward, done, False, {}

    def calculate_reward(self, action):
        target_value = self.output[self.current_batch, self.current_step]
        predicted_value = action
        reward = 1 if predicted_value == target_value else 0
        return reward
