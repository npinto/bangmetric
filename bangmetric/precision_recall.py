"""Precision and Recall
http://en.wikipedia.org/wiki/Precision_and_recall
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['precision', 'recall', 'average_precision']


import numpy as np
from scipy.integrate import trapz

DTYPE = np.float64


def precision(y_true, y_pred):
    """Computes the Precision values w.r.t. descending `y_pred` values.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values.

    y_pred: array, shape = [n_samples]
        Predicted values.

    Returns
    -------
    prec: array, shape = [n_samples]
        Precision array.
    """

    assert len(y_true) == len(y_pred)

    y_true = np.array(y_true, dtype=DTYPE)
    y_pred = np.array(y_pred, dtype=DTYPE)

    idx = (-y_pred).argsort()

    tp = (y_true[idx] > 0).cumsum()
    fp = (y_true[idx] < 0).cumsum()

    prec = tp / (fp + tp)

    return prec


def recall(y_true, y_pred):
    """Computes the Recall values w.r.t. descending `y_pred` values.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values.

    y_pred: array, shape = [n_samples]
        Predicted values.

    Returns
    -------
    rec: array, shape = [n_samples]
        Recall array.
    """

    assert len(y_true) == len(y_pred)

    y_true = np.array(y_true, dtype=DTYPE)
    y_pred = np.array(y_pred, dtype=DTYPE)

    idx = (-y_pred).argsort()

    tp = (y_true[idx] > 0).cumsum()

    rec = tp / (y_true > 0).sum()

    return rec


def average_precision(y_true, y_pred, integration='trapz'):
    """Computes the Average Precision (AP) from the recall and precision
    arrays. Different 'integration' methods can be used.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values.

    y_pred: array, shape = [n_samples]
        Predicted values.

    integration: str
        Type of 'integration' method used to compute the average precision:
            'trapz': trapezoidal rule (default)
            'voc2010': see http://goo.gl/glxdO and http://goo.gl/ueXzr
            'voc2007': see http://goo.gl/E1YyY

    Returns
    -------
    ap: float
        Average Precision
    """

    assert len(y_true) == len(y_pred)
    assert integration in ['trapz', 'voc2010', 'voc2007']

    y_true = np.array(y_true, dtype=DTYPE)
    y_pred = np.array(y_pred, dtype=DTYPE)

    rec = recall(y_true, y_pred)
    prec = precision(y_true, y_pred)

    if integration == 'trapz':
        if rec[0] != 0.:
            rec = np.concatenate(([0.], rec))
            prec = np.concatenate(([prec[0]], prec))
        ap = trapz(prec, rec)

    elif integration == 'voc2010':
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        rng = np.arange(len(mpre) - 1)[::-1]
        for i in rng:
            mpre[i] = max(mpre[i], mpre[i + 1])
        sel = np.nonzero(mrec[1:] != mrec[0:-1])[0] + 1
        ap = ((mrec[sel] - mrec[sel - 1]) * mpre[sel]).sum()
        if np.isnan(ap):
            ap = 0.

    elif integration == 'voc2007':
        ap = 0.
        rng = np.arange(0, 1.1, .1)
        for th in rng:
            p = prec[rec >= th]
            if len(p) > 0:
                ap += p.max() / rng.size

    return ap
