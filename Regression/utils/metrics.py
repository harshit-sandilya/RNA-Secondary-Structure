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
from concurrent.futures import ThreadPoolExecutor


# Define individual metric functions
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


def mse_metric(actual, predicted):
    return mean_squared_error(actual, predicted)


def mae_metric(actual, predicted):
    return mean_absolute_error(actual, predicted)


def r2_metric(actual, predicted):
    return r2_score(actual, predicted)


def calc(predicted, actual):
    rounded_predicted = torch.round(predicted).clamp(min=0, max=3).flatten().numpy()
    actual = actual.flatten().numpy()
    predicted = predicted.flatten().numpy()

    labels = [0, 1, 2, 3]

    metrics_funcs = {
        "accuracy": lambda: accuracy_metric(actual, rounded_predicted),
        "balanced_accuracy": lambda: balanced_accuracy_metric(
            actual, rounded_predicted
        ),
        "f1_score": lambda: f1_metric(actual, rounded_predicted, labels),
        "fbeta_score": lambda: fbeta_metric(actual, rounded_predicted, labels),
        "matthews_corrcoef": lambda: matthews_metric(actual, rounded_predicted),
        "precision_score": lambda: precision_metric(actual, rounded_predicted, labels),
        "recall_score": lambda: recall_metric(actual, rounded_predicted, labels),
        "mean_squared_error_round": lambda: mse_metric(actual, rounded_predicted),
        "mean_absolute_error_round": lambda: mae_metric(actual, rounded_predicted),
        "r2_score_round": lambda: r2_metric(actual, rounded_predicted),
        "mean_squared_error": lambda: mse_metric(actual, predicted),
        "mean_absolute_error": lambda: mae_metric(actual, predicted),
        "r2_score": lambda: r2_metric(actual, predicted),
    }

    with ThreadPoolExecutor() as executor:
        futures = {name: executor.submit(func) for name, func in metrics_funcs.items()}
        results = {name: future.result() for name, future in futures.items()}

    actual_unpad = [actual[i] - 1 for i in range(len(actual)) if actual[i] != 0]
    predicted_unpad = [
        rounded_predicted[i] - 1
        for i in range(len(rounded_predicted))
        if actual[i] != 0
    ]

    actual_unpad = torch.tensor(actual_unpad).numpy()
    predicted_unpad = torch.tensor(predicted_unpad).numpy()
    rounded_predicted_unpad = (
        torch.round(torch.tensor(predicted_unpad)).clamp(min=0, max=2).numpy()
    )
    labels_unpad = [0, 1, 2]

    metrics_funcs_unpad = {
        "accuracy_unpad": lambda: accuracy_metric(
            actual_unpad, rounded_predicted_unpad
        ),
        "balanced_accuracy_unpad": lambda: balanced_accuracy_metric(
            actual_unpad, rounded_predicted_unpad
        ),
        "f1_score_unpad": lambda: f1_metric(
            actual_unpad, rounded_predicted_unpad, labels_unpad
        ),
        "fbeta_score_unpad": lambda: fbeta_metric(
            actual_unpad, rounded_predicted_unpad, labels_unpad
        ),
        "matthews_corrcoef_unpad": lambda: matthews_metric(
            actual_unpad, rounded_predicted_unpad
        ),
        "precision_score_unpad": lambda: precision_metric(
            actual_unpad, rounded_predicted_unpad, labels_unpad
        ),
        "recall_score_unpad": lambda: recall_metric(
            actual_unpad, rounded_predicted_unpad, labels_unpad
        ),
        "mean_squared_error_unpad": lambda: mse_metric(actual_unpad, predicted_unpad),
        "mean_absolute_error_unpad": lambda: mae_metric(actual_unpad, predicted_unpad),
        "r2_score_unpad": lambda: r2_metric(actual_unpad, predicted_unpad),
        "mean_squared_error_unpad_round": lambda: mse_metric(
            actual_unpad, rounded_predicted_unpad
        ),
        "mean_absolute_error_unpad_round": lambda: mae_metric(
            actual_unpad, rounded_predicted_unpad
        ),
        "r2_score_unpad_round": lambda: r2_metric(
            actual_unpad, rounded_predicted_unpad
        ),
    }

    with ThreadPoolExecutor() as executor:
        futures_unpad = {
            name: executor.submit(func) for name, func in metrics_funcs_unpad.items()
        }
        results_unpad = {
            name: future.result() for name, future in futures_unpad.items()
        }

    results.update(results_unpad)
    return results
