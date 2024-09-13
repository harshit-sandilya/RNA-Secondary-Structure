import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold

from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier

from utils.metrics import calc


train_file = "./data/final/train.csv"
test_file = "./data/final/test.csv"


def write_res(writer, model, metric, i):
    writer.writerow(
        {
            "Model": model,
            "Fold": i,
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
X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)
# Y_test = torch.tensor(Y_test)
# print(Y_train.shape[0])
print("Data Loaded")

classes = [0, 1, 2, 3]
models = [
    {
        "name": "Logistic Regression",
        "model": MultiOutputClassifier(OneVsRestClassifier(LogisticRegression())),
    },
    {
        "name": "Ridge Classifier",
        "model": MultiOutputClassifier(RidgeClassifier()),
    },
    {
        "name": "SGD Classifier",
        "model": MultiOutputClassifier(OneVsRestClassifier(SGDClassifier())),
    },
    {
        "name": "Passive Aggressive Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(PassiveAggressiveClassifier())
        ),
    },
    {
        "name": "K Neighbors Classifier",
        "model": KNeighborsClassifier(),
    },
    {
        "name": "SVC",
        "model": MultiOutputClassifier(OneVsRestClassifier(SVC())),
    },
    {
        "name": "Gaussian NB",
        "model": MultiOutputClassifier(GaussianNB()),
    },
    {
        "name": "Multinomial NB",
        "model": MultiOutputClassifier(MultinomialNB()),
    },
    {
        "name": "Bernoulli NB",
        "model": MultiOutputClassifier(BernoulliNB()),
    },
    {
        "name": "Decision Tree Classifier",
        "model": DecisionTreeClassifier(),
    },
    {
        "name": "Random Forest Classifier",
        "model": RandomForestClassifier(),
    },
    {
        "name": "Extra Trees Classifier",
        "model": ExtraTreesClassifier(),
    },
    {
        "name": "Gradient Boosting Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(GradientBoostingClassifier())
        ),
    },
    {
        "name": "Ada Boost Classifier",
        "model": MultiOutputClassifier(AdaBoostClassifier()),
    },
    {
        "name": "Bagging Classifier",
        "model": MultiOutputClassifier(BaggingClassifier()),
    },
    {
        "name": "Linear Discriminant Analysis",
        "model": MultiOutputClassifier(LinearDiscriminantAnalysis()),
    },
    {
        "name": "Gaussian Process Classifier",
        "model": MultiOutputClassifier(
            OneVsRestClassifier(GaussianProcessClassifier())
        ),
    },
]

with open("./logs/resultsML.csv", "w") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Model",
            "Fold",
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

    for model in tqdm(models):
        cls = model["model"]
        for i in range(2, 11):
            kf = KFold(n_splits=i, shuffle=True, random_state=42)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                cls.fit(X_train, Y_train)
                Y_pred = cls.predict(X_test)
                Y_pred = torch.tensor(Y_pred)
                metric = calc(Y_pred, Y_test)
                write_res(writer, model["name"], metric, i)

print("Done")
