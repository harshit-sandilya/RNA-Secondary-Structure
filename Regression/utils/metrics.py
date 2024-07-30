import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def calc(predicted, actual):
    rounded_predicted = torch.round(predicted).clamp(min=0, max=3)
    rounded_predicted = rounded_predicted.flatten()
    actual = actual.flatten()
    predicted = predicted.flatten()

    labels = [0, 1, 2, 3]

    acc = accuracy_score(actual, rounded_predicted)
    bal_acc = balanced_accuracy_score(actual, rounded_predicted)
    f1 = f1_score(
        actual, rounded_predicted, average="macro", labels=labels, zero_division=1
    )
    fbeta = fbeta_score(
        actual,
        rounded_predicted,
        beta=0.5,
        average="macro",
        labels=labels,
        zero_division=1,
    )
    matthews = matthews_corrcoef(actual, rounded_predicted)
    precision = precision_score(
        actual, rounded_predicted, average="macro", labels=labels, zero_division=1
    )
    recall = recall_score(
        actual, rounded_predicted, average="macro", labels=labels, zero_division=1
    )

    mse_round = mean_squared_error(actual, rounded_predicted)
    mae_round = mean_absolute_error(actual, rounded_predicted)
    r2_round = r2_score(actual, rounded_predicted)

    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    actual_unpad = []
    predicted_unpad = []
    for i in range(len(actual)):
        if actual[i] != 0:
            actual_unpad.append(actual[i] - 1)
            predicted_unpad.append(rounded_predicted[i] - 1)

    actual_unpad = torch.tensor(actual_unpad)
    predicted_unpad = torch.tensor(predicted_unpad)
    rounded_predicted_unpad = torch.round(predicted_unpad).clamp(min=0, max=2)
    rounded_predicted_unpad = rounded_predicted_unpad.flatten()

    labels_unpad = [0, 1, 2]

    acc_unpad = accuracy_score(actual_unpad, rounded_predicted_unpad)
    bal_acc_unpad = balanced_accuracy_score(actual_unpad, rounded_predicted_unpad)
    f1_unpad = f1_score(
        actual_unpad,
        rounded_predicted_unpad,
        average="macro",
        labels=labels_unpad,
        zero_division=1,
    )
    fbeta_unpad = fbeta_score(
        actual_unpad,
        rounded_predicted_unpad,
        beta=0.5,
        average="macro",
        labels=labels_unpad,
        zero_division=1,
    )
    matthews_unpad = matthews_corrcoef(actual_unpad, rounded_predicted_unpad)
    precision_unpad = precision_score(
        actual_unpad,
        rounded_predicted_unpad,
        average="macro",
        labels=labels_unpad,
        zero_division=1,
    )
    recall_unpad = recall_score(
        actual_unpad,
        rounded_predicted_unpad,
        average="macro",
        labels=labels_unpad,
        zero_division=1,
    )

    mse_unpad = mean_squared_error(actual_unpad, predicted_unpad)
    mae_unpad = mean_absolute_error(actual_unpad, predicted_unpad)
    r2_unpad = r2_score(actual_unpad, predicted_unpad)

    mse_unpad_round = mean_squared_error(actual_unpad, rounded_predicted_unpad)
    mae_unpad_round = mean_absolute_error(actual_unpad, rounded_predicted_unpad)
    r2_unpad_round = r2_score(actual_unpad, rounded_predicted_unpad)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "fbeta_score": fbeta,
        "matthews_corrcoef": matthews,
        "precision_score": precision,
        "recall_score": recall,
        "mean_squared_error_round": mse_round,
        "mean_absolute_error_round": mae_round,
        "r2_score_round": r2_round,
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2,
        "accuracy_unpad": acc_unpad,
        "balanced_accuracy_unpad": bal_acc_unpad,
        "f1_score_unpad": f1_unpad,
        "fbeta_score_unpad": fbeta_unpad,
        "matthews_corrcoef_unpad": matthews_unpad,
        "precision_score_unpad": precision_unpad,
        "recall_score_unpad": recall_unpad,
        "mean_squared_error_unpad": mse_unpad,
        "mean_absolute_error_unpad": mae_unpad,
        "r2_score_unpad": r2_unpad,
        "mean_squared_error_unpad_round": mse_unpad_round,
        "mean_absolute_error_unpad_round": mae_unpad_round,
        "r2_score_unpad_round": r2_unpad_round,
    }
