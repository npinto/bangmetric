"""Precision and Recall
http://en.wikipedia.org/wiki/Precision_and_recall
"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['precision', 'recall', 'average_precision']


from warnings import warn as warning
import numpy as np
from scipy.integrate import trapz

DTYPE = np.float64


def precision(y_true, y_pred, argsort_kind='quicksort'):
    """Computes the Precision values w.r.t. descending `y_pred` values.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values, interpreted as strictly positive or not
        (i.e. converted to binary).
        Could be in {-1, +1} or {0, 1} or {False, True}.

    y_pred: array, shape = [n_samples]
        Predicted values.

    argsort_kind: str
        Sorting algorithm.

    Returns
    -------
    prec: array, shape = [n_samples]
        Precision array.
    """

    # -- basic checks and conversion
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_true).all()
    assert np.isfinite(y_pred).all()

    y_true = np.array(y_true, dtype=DTYPE)
    assert y_true.ndim == 1

    y_pred = np.array(y_pred, dtype=DTYPE)
    assert y_pred.ndim == 1

    n_uniques = np.unique(y_pred)
    if n_uniques.size == 1:
        raise ValueError('Rank of predicted values is ill-defined'
                         ' because all elements are equal')
    elif n_uniques.size < y_pred.size:
        warning('some predicted elements have exactly the same value.'
                ' output will most probably depend on the sorting'
                ' method used. Here "%s"' % argsort_kind)

    # -- actual computation
    idx = (-y_pred).argsort(kind=argsort_kind)

    tp = (y_true[idx] > 0).cumsum(dtype=DTYPE)
    fp = (y_true[idx] <= 0).cumsum(dtype=DTYPE)

    prec = tp / (fp + tp)

    return prec


def recall(y_true, y_pred, argsort_kind='quicksort'):
    """Computes the Recall values w.r.t. descending `y_pred` values.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values, interpreted as strictly positive or not
        (i.e. converted to binary).
        Could be in {-1, +1} or {0, 1} or {False, True}.

    y_pred: array, shape = [n_samples]
        Predicted values.

    argsort_kind: str
        Sorting algorithm.

    Returns
    -------
    rec: array, shape = [n_samples]
        Recall array.
    """

    # -- basic checks and conversion
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_true).all()
    assert np.isfinite(y_pred).all()

    y_true = np.array(y_true, dtype=DTYPE)
    assert y_true.ndim == 1

    y_pred = np.array(y_pred, dtype=DTYPE)
    assert y_pred.ndim == 1

    n_uniques = np.unique(y_pred)
    if n_uniques.size == 1:
        raise ValueError('Rank of predicted values is ill-defined'
                         ' because all elements are equal')
    elif n_uniques.size < y_pred.size:
        warning('some predicted elements have exactly the same value.'
                ' output will most probably depend on the sorting'
                ' method used. Here "%s"' % argsort_kind)

    # -- actual computation
    idx = (-y_pred).argsort(kind=argsort_kind)

    tp = (y_true[idx] > 0).cumsum(dtype=DTYPE)

    y_true_n_pos = (y_true > 0).sum(dtype=DTYPE)
    if y_true_n_pos == 0:
        rec = np.zeros(tp.shape, dtype=DTYPE)
    else:
        rec = tp / y_true_n_pos

    return rec


def average_precision(y_true, y_pred, integration='trapz',
                      argsort_kind='quicksort'):
    """Computes the Average Precision (AP) from the recall and precision
    arrays. Different 'integration' methods can be used.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values, interpreted as strictly positive or not
        (i.e. converted to binary).
        Could be in {-1, +1} or {0, 1} or {False, True}.

    y_pred: array, shape = [n_samples]
        Predicted values.

    integration: str, optional
        Type of 'integration' method used to compute the average precision:
            'trapz': trapezoidal rule (default)
            'voc2010': see http://goo.gl/glxdO and http://goo.gl/ueXzr
            'voc2007': see http://goo.gl/E1YyY

    argsort_kind: str
        Sorting algorithm.

    Returns
    -------
    ap: float
        Average Precision

    Note
    ----
    'voc2007' method is here only for legacy purposes. We do not recommend
    its use since even simple trivial cases like a perfect match between
    true values and predicted values do not lead to an average precision of 1.
    """

    # -- basic checks and conversion
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_true).all()
    assert np.isfinite(y_pred).all()
    assert integration in ['trapz', 'voc2010', 'voc2007']

    y_true = np.array(y_true, dtype=DTYPE)
    assert y_true.ndim == 1

    y_pred = np.array(y_pred, dtype=DTYPE)
    assert y_pred.ndim == 1

    n_uniques = np.unique(y_pred)
    if n_uniques.size == 1:
        raise ValueError('Rank of predicted values is ill-defined'
                         ' because all elements are equal')
    elif n_uniques.size < y_pred.size:
        warning('some predicted elements have exactly the same value.'
                ' output will most probably depend on the sorting'
                ' method used. Here "%s"' % argsort_kind, UserWarning)

    # -- actual computation
    rec = recall(y_true, y_pred, argsort_kind=argsort_kind)
    prec = precision(y_true, y_pred, argsort_kind=argsort_kind)

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
