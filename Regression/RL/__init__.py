from data import DataModule
from env import Environment
from stable_baselines3 import DDPG

dm = DataModule()
x_train, y_train = dm.train_set()

print(x_train.shape, y_train.shape)

env = Environment(x_train, y_train)
model = DDPG("MlpPolicy", env)
model.learn(total_timesteps=10000, progress_bar=True)
model.save("ddpg_model")
