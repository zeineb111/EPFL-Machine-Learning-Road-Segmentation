import numpy as np
from sklearn.metrics import *


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
            100.0 *
            np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def precision_recall(predictions, labels):
    """ Compute the precision and the recall from dense predictions and 1-hot labels"""
    # linearize matrices
    y_pred = one_hot_to_binary(predictions)
    y_true = one_hot_to_binary(labels)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return precision, recall


def score(preds, labels):
    """ Compute the f1 score from the given precision and recall"""

    return f1_score(labels, preds)


def one_hot_to_binary(array):
    return np.around(np.argmax(array, 1))
