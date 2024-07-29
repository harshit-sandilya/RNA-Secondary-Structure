import torch
from Classification.RL.data import DataModule
from Classification.RL.env import Environment
from stable_baselines3 import DQN
from ..test import calc
from Classification.RL.policy import CustomPolicy
import numpy as np


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


dm = DataModule()
x_test, y_test = dm.test_set()
x_test = normalize(x_test)

for i in range(10):
    print("=====================================")
    env = Environment(x_test, y_test)
    model = DQN.load("dqn_model")

    print("Model loaded")
    total_reward = 0
    predictions = []
    actuals = []

    for batch in range(x_test.shape[0]):
        for step in range(x_test.shape[1]):
            obs = x_test[batch, step]
            obs = obs.astype(np.float32)
            action, _states = model.predict(obs)
            reward = env.calculate_reward(action)
            total_reward += reward
            action = action.item()
            pred = y_test[batch, step][0].item()
            predictions.append(action)
            actuals.append(pred)

    print("Got the predictions")

    predictions = torch.tensor(predictions)
    actuals = torch.tensor(actuals)
    print(f"Total reward: {total_reward}")
    calc(predictions, actuals)
    print("Test complete")
