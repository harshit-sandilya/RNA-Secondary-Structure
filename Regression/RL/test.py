import torch
from Regression.RL.data import DataModule
from Regression.RL.env import Environment
from stable_baselines3 import DQN
from test import calc

dm = DataModule()
x_test, y_test = dm.test_set()

env = Environment(x_test, y_test)
model = DQN("MlpPolicy", env)
model.load("dqn_model")

print("Model loaded")
total_reward = 0
predictions = []
actuals = []

for batch in range(x_test.shape[0]):
    for step in range(x_test.shape[1]):
        obs = x_test[batch, step]
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
