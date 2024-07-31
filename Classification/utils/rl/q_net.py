import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomQNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomQNetwork, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.embed = nn.Embedding(5, 64)
        self.conv1 = nn.Conv1d(64, 16, 15, padding=7)
        self.conv2 = nn.Conv1d(16, 4, 11, padding=5)
        self.conv3 = nn.Conv1d(4, 1, 7, padding=3)
        self.lin = nn.Linear(51, 4)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, -1)
        x = self.lin(x)
        return x


if __name__ == "__main__":
    model = CustomQNetwork(51, 4)
    x = torch.randint(0, 5, (32, 51))
    print(model(x).shape)
