import csv
import torch
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from utils.metrics import calc


train_file = "./data/final/train.csv"
test_file = "./data/final/test.csv"


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
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    df_train = encode(train)
    df_test = encode(test)

    X_train = df_train["sequence"].tolist()
    Y_train = df_train["structure"].tolist()
    X_test = df_test["sequence"].tolist()
    Y_test = df_test["structure"].tolist()

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


X_train, Y_train, X_test, Y_test = load_data()
Y_test = torch.tensor(Y_test)
print(Y_train.shape)
print("Data Loaded")

classes = [0, 1, 2, 3]
models = [
    {
        "name": "Logistic Regression",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(LogisticRegression()), n_jobs=-1
        ),
    },
    {
        "name": "Ridge Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(RidgeClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "SGD Classifier",
        "model": MultiOutputClassifier(OneVsRestClassifier(SGDClassifier()), n_jobs=-1),
    },
    {
        "name": "Passive Aggressive Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(PassiveAggressiveClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "SVC",
        "model": MultiOutputClassifier(OneVsRestClassifier(SVC()), n_jobs=-1),
    },
    {
        "name": "Random Forest Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(RandomForestClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "Decision Tree Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(DecisionTreeClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "Gaussian NB",
        "model": MultiOutputClassifier(OneVsRestClassifier(GaussianNB()), n_jobs=-1),
    },
    {
        "name": "Multinomial NB",
        "model": MultiOutputClassifier(OneVsRestClassifier(MultinomialNB()), n_jobs=-1),
    },
    {
        "name": "Complement NB",
        "model": MultiOutputClassifier(OneVsRestClassifier(ComplementNB()), n_jobs=-1),
    },
    {
        "name": "Bernoulli NB",
        "model": MultiOutputClassifier(OneVsRestClassifier(BernoulliNB()), n_jobs=-1),
    },
    {
        "name": "K Neighbors Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(KNeighborsClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "Gradient Boosting Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(GradientBoostingClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "Ada Boost Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(AdaBoostClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "Bagging Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(BaggingClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "Extra Trees Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(ExtraTreesClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "Random Forest Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(RandomForestClassifier()), n_jobs=-1
        ),
    },
    {
        "name": "Hist Gradient Boosting Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(HistGradientBoostingClassifier()), n_jobs=-1
        ),
    },
]

with open("./logs/regression/resultsML.csv", "w") as f:
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

    for model in models:
        cls = model["model"]
        cls.fit(X_train, Y_train)
        Y_pred = cls.predict(X_test)
        metric = calc(Y_pred, Y_test)
        write_res(writer, model["name"], metric)

print("Done")
