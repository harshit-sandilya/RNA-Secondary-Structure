import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset


class DataModule:
    def __init__(self):
        self.max_length = 100
        self.min_length = 30

    def _convert_to_tensor(self, df):
        t_x = df["sequence"].tolist()
        t_y = df["structure"].tolist()
        for i in range(len(t_x)):
            t_x[i] = torch.tensor(t_x[i], dtype=torch.long)
            t_y[i] = torch.tensor(t_y[i], dtype=torch.long)
            left = math.floor((self.max_length - len(t_x[i])) / 2)
            right = math.ceil((self.max_length - len(t_x[i])) / 2)
            t_x[i] = torch.cat(
                [
                    torch.zeros(left, dtype=torch.long),
                    t_x[i],
                    torch.zeros(right, dtype=torch.long),
                ]
            )
            t_y[i] = torch.cat(
                [
                    torch.zeros(left, dtype=torch.long),
                    t_y[i],
                    torch.zeros(right, dtype=torch.long),
                ]
            )
            t_x[i] = torch.cat(
                [
                    torch.zeros(15, dtype=torch.long),
                    t_x[i],
                    torch.zeros(15, dtype=torch.long),
                ]
            )
            t_y[i] = torch.cat(
                [
                    torch.zeros(15, dtype=torch.long),
                    t_y[i],
                    torch.zeros(15, dtype=torch.long),
                ]
            )
            t_x[i] = t_x[i].unfold(0, 31, 1)
            t_y[i] = t_y[i].unfold(0, 31, 1)
        x = torch.stack(t_x)
        y = torch.stack(t_y)
        y = y[:, :, 15:16]
        return x, y

    def _process_df(self, df):
        df = df[["sequence", "structure"]]
        df = df[df["sequence"].str.len() <= self.max_length]
        df = df[df["sequence"].str.len() >= self.min_length]
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
        self.x_train = self.train_x.numpy()
        self.y_train = self.train_y.numpy()
        return self.x_train, self.y_train

    def test_set(self):
        self.test_file = "./data/final/test.csv"
        dftest = pd.read_csv(self.test_file)
        self.test_x, self.test_y = self._process_df(dftest)
        self.x_test = self.test_x.numpy()
        self.y_test = self.test_y.numpy()
        return self.x_test, self.y_test
