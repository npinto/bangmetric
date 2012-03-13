"""Accuracy"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['accuracy']

import numpy as np


def accuracy(y_true, y_pred, balanced=False):
    """Computes the Accuracy of the predictions (also known as the
    zero-one score).

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values, interpreted as strictly positive or not
        (i.e. converted to binary).

    y_pred: array, shape = [n_samples]
        Predicted values, interpreted as strictly positive or not
        (i.e. converted to binary).

    balanced: bool, optional
        Returns the balanced accuracy (equal weight for positive and
        negative predictions).

    Returns
    -------
    acc: float
        Accuracy (zero-one score).
    """

    assert len(y_true) == len(y_pred)

    # -- "binarize" the arguments
    y_true = np.array(y_true) > 0
    y_pred = np.array(y_pred) > 0

    if balanced:
        pos = y_true > 0
        neg = ~pos
        pos_acc = (y_true[pos] == y_pred[pos]).mean()
        neg_acc = (y_true[neg] == y_pred[neg]).mean()
        acc = (pos_acc + neg_acc) / 2.0
    else:
        acc = (y_true == y_pred).mean()

    return acc
