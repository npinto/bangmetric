"""Correlations (Pearson, Spearman)"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

# TODO: add argsort_kind='quicksort' kwarg

__all__ = ['pearson', 'spearman']

import numpy as np
from numpy import linalg as la

DTYPE = np.float64


def pearson(y_true, y_pred):
    """Computes the Pearson Correlation coefficient between the
    predicted values `y_pred` and ground truth values `y_true`.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values.

    y_pred: array, shape = [n_samples]
        Predicted values.

    Returns
    -------
    phi: float
        Pearson Correlation coefficient (sometimes called "phi
        coefficient" or "Matthews correlation coefficient")

    References
    ----------
    http://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    http://en.wikipedia.org/wiki/Phi_coefficient
    http://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    """

    # -- basic checks and conversion
    assert len(y_true) == len(y_pred)

    y_true = np.array(y_true, dtype=DTYPE)
    assert y_true.ndim == 1

    y_pred = np.array(y_pred, dtype=DTYPE)
    assert y_pred.ndim == 1

    # -- actual computation
    y_true -= y_true.mean()
    y_true_norm = la.norm(y_true)
    assert y_true_norm != 0
    y_true /= y_true_norm

    y_pred -= y_pred.mean()
    y_pred_norm = la.norm(y_pred)
    assert y_pred_norm != 0
    y_pred /= y_pred_norm

    rho = np.dot(y_true, y_pred)

    return rho


def spearman(y_true, y_pred, argsort_kind='quicksort'):
    """Computes the Spearman's rank Correlation coefficient between the
    predicted values `y_pred` and ground truth values `y_true`.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values.

    y_pred: array, shape = [n_samples]
        Predicted values.

    argsort_kind: {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm (see `numpy.argsort`).

    Returns
    -------
    rho: float
        Spearman's rank Correlation coefficient (sometimes called "rho").

    References
    ----------
    http://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient
    """

    # -- basic checks and conversion
    assert len(y_true) == len(y_pred)

    y_true = np.array(y_true, dtype=DTYPE)
    assert y_true.ndim == 1

    y_pred = np.array(y_pred, dtype=DTYPE)
    assert y_pred.ndim == 1

    # -- actual computation
    r_true = np.empty_like(y_true)
    r_true[y_true.argsort(kind=argsort_kind)] = np.arange(y_true.size)
    r_pred = np.empty_like(y_pred)
    r_pred[y_pred.argsort(kind=argsort_kind)] = np.arange(y_pred.size)

    r_true -= r_true.mean()
    r_true_norm = np.linalg.norm(r_true)
    assert r_true_norm != 0
    r_true /= r_true_norm

    r_pred -= r_pred.mean()
    r_pred_norm = np.linalg.norm(r_pred)
    assert r_pred_norm != 0
    r_pred /= r_pred_norm

    rho = np.dot(r_true.ravel(), r_pred.ravel())

    return rho
