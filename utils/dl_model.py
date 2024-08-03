import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.metrics import calc


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(5, 128)
        self.linear = nn.Linear(128, 4)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        self.log_dict(calc(y_hat, y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
