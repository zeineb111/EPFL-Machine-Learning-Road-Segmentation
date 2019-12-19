import numpy as np


def f1_scores(history):
    """ Compute an array of training f1_scores from the epoch history of the model"""

    # Retrieve the values for precision and recall for each epoch
    precision = np.array(history.history['precision'])
    recall = np.array(history.history['recall'])

    return 2 * precision * recall / (precision + recall)
