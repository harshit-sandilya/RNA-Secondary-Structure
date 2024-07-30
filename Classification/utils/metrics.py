import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def calc(predicted, actual):
    predicted = predicted.flatten()
    actual = actual.flatten()
    labels = [0, 1, 2, 3]

    acc = accuracy_score(actual, predicted)
    bal_acc = balanced_accuracy_score(actual, predicted)
    f1 = f1_score(actual, predicted, average="macro", labels=labels, zero_division=1)
    fbeta = fbeta_score(
        actual,
        predicted,
        beta=0.5,
        average="macro",
        labels=labels,
        zero_division=1,
    )
    matthews = matthews_corrcoef(actual, predicted)
    precision = precision_score(
        actual, predicted, average="macro", labels=labels, zero_division=1
    )
    recall = recall_score(
        actual, predicted, average="macro", labels=labels, zero_division=1
    )

    actual_unpad = []
    predicted_unpad = []
    for i in range(len(actual)):
        if actual[i] != 0:
            actual_unpad.append(actual[i] - 1)
            predicted_unpad.append(predicted[i] - 1)

    actual_unpad = torch.tensor(actual_unpad)
    predicted_unpad = torch.tensor(predicted_unpad)

    labels_unpad = [0, 1, 2]

    acc_unpad = accuracy_score(actual_unpad, predicted_unpad)
    bal_acc_unpad = balanced_accuracy_score(actual_unpad, predicted_unpad)
    f1_unpad = f1_score(
        actual_unpad,
        predicted_unpad,
        average="macro",
        labels=labels_unpad,
        zero_division=1,
    )
    fbeta_unpad = fbeta_score(
        actual_unpad,
        predicted_unpad,
        beta=0.5,
        average="macro",
        labels=labels_unpad,
        zero_division=1,
    )
    matthews_unpad = matthews_corrcoef(actual_unpad, predicted_unpad)
    precision_unpad = precision_score(
        actual_unpad,
        predicted_unpad,
        average="macro",
        labels=labels_unpad,
        zero_division=1,
    )
    recall_unpad = recall_score(
        actual_unpad,
        predicted_unpad,
        average="macro",
        labels=labels_unpad,
        zero_division=1,
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "fbeta_score": fbeta,
        "matthews_corrcoef": matthews,
        "precision_score": precision,
        "recall_score": recall,
        "accuracy_unpad": acc_unpad,
        "balanced_accuracy_unpad": bal_acc_unpad,
        "f1_score_unpad": f1_unpad,
        "fbeta_score_unpad": fbeta_unpad,
        "matthews_corrcoef_unpad": matthews_unpad,
        "precision_score_unpad": precision_unpad,
        "recall_score_unpad": recall_unpad,
    }
