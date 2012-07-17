"""D' (d-prime) Sensitivity Index"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

__all__ = ['dprime']

import numpy as np


def dprime(y_pred, y_true):
    """Computes the d-prime sensitivity index of the predictions.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values, interpreted as strictly positive or not
        (i.e. converted to binary).
        Could be in {-1, +1} or {0, 1} or {False, True}.

    y_pred: array, shape = [n_samples]
        Predicted values (real).

    Returns
    -------
    dp: float or None
        d-prime, None if d-prime is undefined

    References
    ----------
    http://en.wikipedia.org/wiki/D'
    """

    # -- basic checks and conversion
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_true).all()
    assert np.isfinite(y_pred).all()

    y_true = np.array(y_true)
    assert y_true.ndim == 1

    y_pred = np.array(y_pred)
    assert y_pred.ndim == 1

    # -- actual computation
    pos = y_true > 0
    neg = ~pos
    pos_mean = y_pred[pos].mean()
    neg_mean = y_pred[neg].mean()
    pos_var = y_pred[pos].var(ddof=1)
    neg_var = y_pred[neg].var(ddof=1)

    num = pos_mean - neg_mean
    div = np.sqrt((pos_var + neg_var) / 2.)
    if div == 0:
        dp = None
    else:
        dp = num / div

    return dp
