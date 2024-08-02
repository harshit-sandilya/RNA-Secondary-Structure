import torch
import time
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from utils.rl_data import DataModule
from utils.rl_env import Environment
from utils.metrics import calc


start_time = time.time()


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

model = DQN("MlpPolicy", env, tensorboard_log="./logs/dqn/")
model.learn(total_timesteps=100000, progress_bar=True, callback=TensorboardCallback())
model.save("dqn_model")
print(f"[{time.time()-start_time}]Model trained")


x_test, y_test = dm.test_set()
x_test = normalize(x_test)

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
print(f"[{time.time()-start_time}]Got the predictions")

predictions = torch.tensor(predictions)
actuals = torch.tensor(actuals)
print(f"Total reward: {total_reward}")
print(calc(predictions, actuals))
print(f"[{time.time()-start_time}]Test complete")
