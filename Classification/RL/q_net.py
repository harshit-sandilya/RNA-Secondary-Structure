import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomQNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.fc = nn.Linear(64 * 27, 4)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x
