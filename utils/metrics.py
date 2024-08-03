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
from concurrent.futures import ThreadPoolExecutor


def accuracy_metric(actual, predicted):
    return accuracy_score(actual, predicted)


def balanced_accuracy_metric(actual, predicted):
    return balanced_accuracy_score(actual, predicted)


def f1_metric(actual, predicted, labels):
    return f1_score(actual, predicted, average="macro", labels=labels, zero_division=1)


def fbeta_metric(actual, predicted, labels):
    return fbeta_score(
        actual, predicted, beta=0.5, average="macro", labels=labels, zero_division=1
    )


def matthews_metric(actual, predicted):
    return matthews_corrcoef(actual, predicted)


def precision_metric(actual, predicted, labels):
    return precision_score(
        actual, predicted, average="macro", labels=labels, zero_division=1
    )


def recall_metric(actual, predicted, labels):
    return recall_score(
        actual, predicted, average="macro", labels=labels, zero_division=1
    )


def calc(predicted, actual):
    if predicted.device.type != "cpu":
        predicted = predicted.cpu()
    if actual.device.type != "cpu":
        actual = actual.cpu()
    predicted = predicted.flatten().numpy()
    actual = actual.flatten().numpy()
    labels = [0, 1, 2, 3]

    metrics_funcs = {
        "accuracy": accuracy_metric,
        "balanced_accuracy": balanced_accuracy_metric,
        "f1_score": f1_metric,
        "fbeta_score": fbeta_metric,
        "matthews_corrcoef": matthews_metric,
        "precision_score": precision_metric,
        "recall_score": recall_metric,
    }

    with ThreadPoolExecutor() as executor:
        futures = {
            name: (
                executor.submit(func, actual, predicted, labels)
                if "labels" in func.__code__.co_varnames
                else executor.submit(func, actual, predicted)
            )
            for name, func in metrics_funcs.items()
        }

        results = {name: future.result() for name, future in futures.items()}

    actual_unpad = [actual[i] - 1 for i in range(len(actual)) if actual[i] != 0]
    predicted_unpad = [
        predicted[i] - 1 for i in range(len(predicted)) if actual[i] != 0
    ]

    actual_unpad = torch.tensor(actual_unpad)
    predicted_unpad = torch.tensor(predicted_unpad)
    labels_unpad = [0, 1, 2]

    with ThreadPoolExecutor() as executor:
        futures_unpad = {
            "accuracy_unpad": executor.submit(
                accuracy_metric, actual_unpad, predicted_unpad
            ),
            "balanced_accuracy_unpad": executor.submit(
                balanced_accuracy_metric, actual_unpad, predicted_unpad
            ),
            "f1_score_unpad": executor.submit(
                f1_metric, actual_unpad, predicted_unpad, labels_unpad
            ),
            "fbeta_score_unpad": executor.submit(
                fbeta_metric, actual_unpad, predicted_unpad, labels_unpad
            ),
            "matthews_corrcoef_unpad": executor.submit(
                matthews_metric, actual_unpad, predicted_unpad
            ),
            "precision_score_unpad": executor.submit(
                precision_metric, actual_unpad, predicted_unpad, labels_unpad
            ),
            "recall_score_unpad": executor.submit(
                recall_metric, actual_unpad, predicted_unpad, labels_unpad
            ),
        }

        results_unpad = {
            name: future.result() for name, future in futures_unpad.items()
        }

    results.update(results_unpad)
    return results
