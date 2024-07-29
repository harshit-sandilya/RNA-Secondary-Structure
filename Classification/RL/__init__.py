import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from data import DataModule
from env import Environment
from policy import CustomPolicy


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record("train/reward", self.locals["rewards"])
        return True


dm = DataModule()
x_train, y_train = dm.train_set()
x_train = normalize(x_train)

env = Environment(x_train, y_train)

# model = DQN(CustomPolicy, env, tensorboard_log="./logs/Classification/dqn/")
model = DQN("MlpPolicy", env, tensorboard_log="./logs/Classification/dqn/")
model.learn(total_timesteps=100000, progress_bar=True, callback=TensorboardCallback())
model.save("dqn_model")
