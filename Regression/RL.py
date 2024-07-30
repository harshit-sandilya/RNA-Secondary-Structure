import torch
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback

from utils.rl.env import Environment
from utils.rl.data import DataModule
from utils.metrics import calc


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        self.logger.record("train/reward", self.locals["rewards"])
        return True


dm = DataModule()
x_train, y_train = dm.train_set()
x_train = normalize(x_train)
env = Environment(x_train, y_train)

model = DDPG("MlpPolicy", env, tensorboard_log="./logs/regression/ddpg/")
model.learn(total_timesteps=10000, progress_bar=True, callback=TensorboardCallback())
model.save("ddpg_model")
print("Model trained")


x_test, y_test = dm.test_set()
x_test = normalize(x_test)

total_reward = 0
predictions = []
actuals = []

for batch in range(x_test.shape[0]):
    for step in range(x_test.shape[1]):
        obs = x_test[batch, step]
        action, _states = model.predict(obs)
        reward = env.calculate_reward(action)
        total_reward += reward
        pred = y_test[batch, step]
        predictions.append(action)
        actuals.append(pred)
print("Got the predictions")

actuals = torch.tensor(actuals)
predictions = torch.tensor(predictions)
print(f"Total reward: {total_reward}")
print(calc(predictions, actuals))
print("Test complete")
