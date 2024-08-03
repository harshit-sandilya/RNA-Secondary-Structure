import torch
import torch.nn as nn
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(5, 128)
        self.conv1 = nn.Conv1d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 8, 3, padding=1)
        self.linear = nn.Linear(8, 4)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
