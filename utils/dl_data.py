import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(self):
        super(DataModule, self).__init__()
        self.max_length = 300
        self.min_length = 30

    def _convert_to_tensor(self, df):
        x = df["sequence"].tolist()
        y = df["structure"].tolist()
        for i in range(len(x)):
            x[i] = torch.tensor(
                x[i] + [0] * (self.max_length - len(x[i])), dtype=torch.long
            )
            y[i] = torch.tensor(
                y[i] + [0] * (self.max_length - len(y[i])), dtype=torch.long
            )
        x = torch.stack(x)
        y = torch.stack(y)
        return x, y

    def _process_df(self, df):
        df = df[["sequence", "structure"]]
        df = df[df["sequence"].str.len() <= self.max_length]
        df = df[df["sequence"].str.len() >= self.min_length]
        df = df[df["structure"].str.len() == df["sequence"].str.len()]
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=["sequence"])
        df = df.reset_index(drop=True)
        df = df[df["sequence"].str.contains("^[AGUC]+$")]
        df = df[df["structure"].str.contains("^[/./).(]+$")]
        encoding_dict = {"A": 1, "C": 2, "G": 3, "U": 4}
        decoding_dict = {".": 1, "(": 2, ")": 3}
        df["sequence"] = df["sequence"].apply(
            lambda x: [encoding_dict[base] for base in x]
        )
        df["structure"] = df["structure"].apply(
            lambda x: [decoding_dict[base] for base in x]
        )
        x, y = self._convert_to_tensor(df)
        return x, y

    def train_set(self):
        self.train_file = "./data/final/train.csv"
        dftrain = pd.read_csv(self.train_file)
        self.train_x, self.train_y = self._process_df(dftrain)
        self.x_train = self.train_x
        self.y_train = self.train_y
        return self.x_train, self.y_train

    def test_set(self):
        self.test_file = "./data/final/test.csv"
        dftest = pd.read_csv(self.test_file)
        self.test_x, self.test_y = self._process_df(dftest)
        self.x_test = self.test_x
        self.y_test = self.test_y
        return self.x_test, self.y_test

    def setup(self, stage=None):
        self.train_set()
        self.test_set()
        self.train_data = TensorDataset(self.x_train, self.y_train)
        self.test_data = TensorDataset(self.x_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=32)
