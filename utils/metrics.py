import numpy as np


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
            100.0 *
            np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def precision_recall(predictions, labels):
    """ Compute the precision and the recall from dense predictions and 1-hot labels"""
    n_fp = 1.0  # False postives
    n_fn = 1.0  # False negatives
    n_tp = 1.0  # True positives
    n_tn = 1.0  # True negatives
    pred = np.around(predictions)
    pred_label_stack = np.column_stack((pred, labels))

    for row in pred_label_stack:
        if ~((row - [0, 1, 1, 0]).any()):
            n_fp += 1
        elif ~((row - [1, 0, 0, 1]).any()):
            n_fn += 1
        elif ~((row - [0, 1, 0, 1]).any()):
            n_tp += 1
        else:
            n_tn += 1

    precision = n_tp / (n_tp + n_fp)
    recall = n_tp / (n_tp + n_fn)

    return precision, recall


def f1_score(precision, recall):
    """ Compute the f1 score from the given precision and recall"""

    return 2 * precision * recall / (precision + recall)
