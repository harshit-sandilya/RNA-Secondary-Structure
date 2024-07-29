from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    SGDRegressor,
    ElasticNet,
    Lars,
    Lasso,
    LassoLars,
    OrthogonalMatchingPursuit,
    ARDRegression,
    BayesianRidge,
    HuberRegressor,
    QuantileRegressor,
    TheilSenRegressor,
    PoissonRegressor,
    TweedieRegressor,
    PassiveAggressiveRegressor,
)
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from test import calc
from tqdm import tqdm
import pandas as pd
import numpy as np
import csv
import torch


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
            "Mean Squared Error": round(metric["mean_squared_error"], 2),
            "Mean Absolute Error": round(metric["mean_absolute_error"], 2),
            "R2 Score": round(metric["r2_score"], 2),
            "Mean Squared Error Rounded": round(metric["mean_squared_error_round"], 2),
            "Mean Absolute Error Rounded": round(
                metric["mean_absolute_error_round"], 2
            ),
            "R2 Score Rounded": round(metric["r2_score_round"], 2),
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
            "Mean Squared Error Unpadded": round(metric["mean_squared_error_unpad"], 2),
            "Mean Absolute Error Unpadded": round(
                metric["mean_absolute_error_unpad"], 2
            ),
            "R2 Score Unpadded": round(metric["r2_score_unpad"], 2),
            "Mean Squared Error Unpadded Rounded": round(
                metric["mean_squared_error_unpad_round"], 2
            ),
            "Mean Absolute Error Unpadded Rounded": round(
                metric["mean_absolute_error_unpad_round"], 2
            ),
            "R2 Score Unpadded Rounded": round(metric["r2_score_unpad_round"], 2),
        }
    )


def encode(df):
    df = df[df["length"] <= 100]
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
print("Data Loaded")

linear_models = [
    {
        "name": "Linear Regression",
        "model": LinearRegression(n_jobs=-1),
    },
    {
        "name": "Ridge Regression",
        "model": Ridge(alpha=0.1, random_state=0),
    },
    {
        "name": "Elastic Net",
        "model": ElasticNet(alpha=0.1, random_state=0),
    },
    {
        "name": "Least Angle Regressor",
        "model": Lars(random_state=0, eps=1e-3),
    },
    {
        "name": "Lasso Regression",
        "model": Lasso(alpha=0.1, random_state=0),
    },
    {
        "name": "Lasso Lars",
        "model": LassoLars(alpha=0.1, random_state=0),
    },
    {
        "name": "Orthogonal Matching Pursuit",
        "model": OrthogonalMatchingPursuit(),
    },
    {
        "name": "SGD Regressor",
        "model": MultiOutputRegressor(
            SGDRegressor(max_iter=1000, alpha=0.1), n_jobs=-1
        ),
    },
    {
        "name": "ARD Regression",
        "model": MultiOutputRegressor(ARDRegression(), n_jobs=-1),
    },
    {
        "name": "Bayesian Ridge",
        "model": MultiOutputRegressor(BayesianRidge(), n_jobs=-1),
    },
    {
        "name": "Huber Regressor",
        "model": MultiOutputRegressor(HuberRegressor(max_iter=1000), n_jobs=-1),
    },
    {
        "name": "Quantile Regressor",
        "model": MultiOutputRegressor(QuantileRegressor(alpha=0.1), n_jobs=8),
    },
    {
        "name": "Theil Sen Regressor",
        "model": MultiOutputRegressor(TheilSenRegressor(random_state=0), n_jobs=8),
    },
    {
        "name": "Poisson Regressor",
        "model": MultiOutputRegressor(PoissonRegressor(max_iter=1000), n_jobs=-1),
    },
    {
        "name": "Tweedie Regressor",
        "model": MultiOutputRegressor(TweedieRegressor(alpha=0.1), n_jobs=-1),
    },
    {
        "name": "Passive Aggressive Regressor",
        "model": MultiOutputRegressor(PassiveAggressiveRegressor(), n_jobs=-1),
    },
]

ensemble_models = [
    {
        "name": "AdaBoost Regressor",
        "model": MultiOutputRegressor(AdaBoostRegressor(), n_jobs=-1),
    },
    {
        "name": "Bagging Regressor",
        "model": MultiOutputRegressor(BaggingRegressor(), n_jobs=-1),
    },
    {
        "name": "Gradient Boosting Regressor",
        "model": MultiOutputRegressor(GradientBoostingRegressor(), n_jobs=-1),
    },
    {
        "name": "Hist Gradient Boosting Regressor",
        "model": MultiOutputRegressor(HistGradientBoostingRegressor(), n_jobs=-1),
    },
    {
        "name": "Random Forest Regressor",
        "model": RandomForestRegressor(n_jobs=-1, random_state=0),
    },
]

svm_models = [
    {
        "name": "SVR",
        "model": MultiOutputRegressor(SVR(max_iter=1000), n_jobs=-1),
    },
    {
        "name": "Linear SVR",
        "model": MultiOutputRegressor(LinearSVR(max_iter=1000), n_jobs=-1),
    },
    {
        "name": "NuSVR",
        "model": MultiOutputRegressor(NuSVR(max_iter=1000), n_jobs=-1),
    },
]

tree_models = [
    {
        "name": "Decision Tree Regressor",
        "model": DecisionTreeRegressor(random_state=0),
    },
    {
        "name": "Extra Trees Regressor",
        "model": ExtraTreeRegressor(random_state=0),
    },
]

with open("./logs/Regression/resultsML.csv", "w") as f:
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
            "Mean Squared Error",
            "Mean Absolute Error",
            "R2 Score",
            "Mean Squared Error Rounded",
            "Mean Absolute Error Rounded",
            "R2 Score Rounded",
            "Accuracy Unpadded",
            "Balanced Accuracy Unpadded",
            "F1 Score Unpadded",
            "Fbeta Score Unpadded",
            "Matthews Correlation Coefficient Unpadded",
            "Precision Score Unpadded",
            "Recall Score Unpadded",
            "Mean Squared Error Unpadded",
            "Mean Absolute Error Unpadded",
            "R2 Score Unpadded",
            "Mean Squared Error Unpadded Rounded",
            "Mean Absolute Error Unpadded Rounded",
            "R2 Score Unpadded Rounded",
        ],
    )
    writer.writeheader()

    print("Training Linear Models:")
    for model in tqdm(linear_models):
        reg = model["model"]
        reg.fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        Y_pred = torch.tensor(Y_pred)
        metric_result = calc(Y_pred, Y_test)
        write_res(writer, model["name"], metric_result)

    print("Training Ensemble Models:")
    for model in tqdm(ensemble_models):
        reg = model["model"]
        reg.fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        Y_pred = torch.tensor(Y_pred)
        metric_result = calc(Y_pred, Y_test)
        write_res(writer, model["name"], metric_result)

    print("Training SVM Models:")
    for model in tqdm(svm_models):
        reg = model["model"]
        reg.fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        Y_pred = torch.tensor(Y_pred)
        metric_result = calc(Y_pred, Y_test)
        write_res(writer, model["name"], metric_result)

    print("Training Tree Models:")
    for model in tqdm(tree_models):
        reg = model["model"]
        reg.fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        Y_pred = torch.tensor(Y_pred)
        metric_result = calc(Y_pred, Y_test)
        write_res(writer, model["name"], metric_result)
