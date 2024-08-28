from sklearn.ensemble import RandomForestClassifier
import torch
import pandas as pd
import csv
import numpy as np
from utils.metrics import calc

train_file = [
    "./data/final/bpRNA-train.csv",
    "./data/final/RNAstrand-train.csv",
    "./data/final/RFAM-train.csv",
]
test_file = "./data/final/RNAstrand-test.csv"


def write_res(writer, model, metric):
    writer.writerow(
        {
            "Model": model,
            "Accuracy": round(metric["accuracy"] * 100, 2),
            "Balanced Accuracy": round(metric["balanced_accuracy"] * 100, 2),
            "F1 Score": round(metric["f1_score"] * 100, 2),
            "Fbeta Score": round(metric["fbeta_score"] * 100, 2),
            "Matthews Correlation Coefficient": round(
                metric["matthews_corrcoef"] * 100, 2
            ),
            "Precision Score": round(metric["precision_score"] * 100, 2),
            "Recall Score": round(metric["recall_score"] * 100, 2),
            "Accuracy Unpadded": round(metric["accuracy_unpad"] * 100, 2),
            "Balanced Accuracy Unpadded": round(
                metric["balanced_accuracy_unpad"] * 100, 2
            ),
            "F1 Score Unpadded": round(metric["f1_score_unpad"] * 100, 2),
            "Fbeta Score Unpadded": round(metric["fbeta_score_unpad"] * 100, 2),
            "Matthews Correlation Coefficient Unpadded": round(
                metric["matthews_corrcoef_unpad"] * 100, 2
            ),
            "Precision Score Unpadded": round(metric["precision_score_unpad"] * 100, 2),
            "Recall Score Unpadded": round(metric["recall_score_unpad"] * 100, 2),
        }
    )


def encode(df):
    df = df[df["length"] <= 300]
    df = df[df["length"] >= 30]
    df = df[["sequence", "structure"]]
    df["sequence"] = df["sequence"].str.upper()
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[df["sequence"].str.contains("^[AGUC]+$")]
    df = df[df["structure"].str.contains("^[/./).(]+$")]
    encoding_dict = {"A": 1, "C": 2, "G": 3, "U": 4}
    decoding_dict = {".": 1, "(": 2, ")": 3}
    df["sequence"] = df["sequence"].apply(lambda x: [encoding_dict[base] for base in x])
    df["structure"] = df["structure"].apply(
        lambda x: [decoding_dict[base] for base in x]
    )
    df["sequence"] = df["sequence"].apply(lambda x: x + [0] * (300 - len(x)))
    df["structure"] = df["structure"].apply(lambda x: x + [0] * (300 - len(x)))
    return df


def load_data():
    train0 = pd.read_csv(train_file[0])
    train1 = pd.read_csv(train_file[1])
    train2 = pd.read_csv(train_file[2])
    test = pd.read_csv(test_file)

    train = pd.concat([train0, train1, train2])
    train.reset_index(drop=True, inplace=True)
    train = train.sample(frac=1).reset_index(drop=True)

    df_train = encode(train)
    df_test = encode(test)

    X_train = df_train["sequence"].tolist()
    Y_train = df_train["structure"].tolist()
    X_test = df_test["sequence"].tolist()
    Y_test = df_test["structure"].tolist()

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


X_train, Y_train, X_test, Y_test = load_data()
Y_test = torch.tensor(Y_test)
print(Y_train.shape[0])
print("Data Loaded")

classes = [0, 1, 2, 3]

with open("./logs/resultsML.csv", "w") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "F1 Score",
            "Fbeta Score",
            "Matthews Correlation Coefficient",
            "Precision Score",
            "Recall Score",
            "Accuracy Unpadded",
            "Balanced Accuracy Unpadded",
            "F1 Score Unpadded",
            "Fbeta Score Unpadded",
            "Matthews Correlation Coefficient Unpadded",
            "Precision Score Unpadded",
            "Recall Score Unpadded",
        ],
    )
    writer.writeheader()

    cls = RandomForestClassifier()
    cls.fit(X_train, Y_train)
    Y_pred = cls.predict(X_test)
    Y_pred = torch.tensor(Y_pred)
    metric = calc(Y_pred, Y_test)
    write_res(writer, "Random_Forest", metric)

print("Done")
