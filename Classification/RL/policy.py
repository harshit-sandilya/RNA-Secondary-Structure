from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from q_net import CustomQNetwork
import torch


class CustomPolicy(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        net_arch=None,
        **kwargs
    ):
        super(CustomPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )
        self.net_arch = net_arch
        self.q_net = CustomQNetwork(observation_space, action_space)
        self.q_net_target = CustomQNetwork(observation_space, action_space)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def _build(self, lr_schedule):
        pass

    def forward(self, obs, deterministic: bool = False):
        return self.q_net(obs)

    def _predict(self, obs, deterministic: bool = False):
        q_values = self.forward(obs)
        actions = torch.argmax(q_values, dim=1).reshape(-1)
        return actions

    def _get_data(self):
        return {
            "net_arch": self.net_arch,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }

    def _load_from_file(self, file_path: str, **kwargs):
        pass

    def _load(self, data, **kwargs):
        pass

    def _save_to_file(self, file_path: str, **kwargs):
        pass
