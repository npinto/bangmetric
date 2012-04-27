"""Root Mean Square Error"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#
# License: BSD

# TODO: rmse_coefficient (i.e. CV(RMSE))
# see http://en.wikipedia.org/wiki/Root-mean-square_deviation

__all__ = ['rmse']

import numpy as np
from numpy import linalg as la

DTYPE = np.float64


def rmse(y_true, y_pred, balanced=False, normalized=False):
    """Computes the Root Mean Square Error (RMSE) between the predicted
    values `y_pred` and ground truth values `y_true`.

    Parameters
    ----------
    y_true: array, shape = [n_samples]
        True values.

    y_pred: array, shape = [n_samples]
        Predicted values.

    balanced: bool, optional (default=False)
        Returns the balanced accuracy (equal weight for positive and
        negative values).

    normalize: bool, optional (default=False)
        Returns the normalized RMSE, i.e. RMSE divided by the range of
        observed values.

    Returns
    -------
    error: float
        RMSE.

    References
    ----------
    http://en.wikipedia.org/wiki/Root-mean-square_deviation
    """

    # -- basic checks and conversion
    assert len(y_true) == len(y_pred)
    assert np.isfinite(y_true).all()
    assert np.isfinite(y_pred).all()

    y_true = np.array(y_true, dtype=DTYPE)
    assert y_true.ndim == 1

    y_pred = np.array(y_pred, dtype=DTYPE)
    assert y_pred.ndim == 1

    # Note that we use la.norm as it is faster than element-wise
    # operations and reductions in numpy when a fast blas library is
    # available

    if balanced:
        pos = y_true > 0
        neg = ~pos
        pos_err = la.norm(y_true[pos] - y_pred[pos])
        pos_err /= np.sqrt(y_true[pos].size)
        neg_err = la.norm(y_true[neg] - y_pred[neg])
        neg_err /= np.sqrt(y_true[neg].size)
        error = (pos_err + neg_err) / 2.
    else:
        error = la.norm(y_true - y_pred)
        error /= np.sqrt(y_true.size)

    if normalized:
        div = y_true.max() - y_true.min()
        if div != 0:
            error /= div

    return error
