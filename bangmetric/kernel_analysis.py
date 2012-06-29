"""Kernel Analysis of a Representation

XXX: description missing


License
-------
Copyright: Gregoire Montavon

This code is released under the MIT licence:
http://www.opensource.org/licenses/mit-license.html


References
----------

* Kernel Analysis of Deep Networks
G. Montavon, M. Braun, K.-R. Mueller
2011, Journal of Machine Learning Research (JMLR)

* On Relevant Dimensions in Kernel Feature Spaces
M. Braun, J. Buhmann, K.-R. Mueller
2008, Journal of Machine Learning Research (JMLR)


Performance
-----------

This code scales up to few thousands samples and dimensions. The
computation of the kernel matrix and eigenvectors for bigger datasets
requires ad-hoc solutions.
@npinto: possibly using random projections, e.g. sklearn's RandomizedPCA

"""

# Authors: Gregoire Montavon <gregoire@montavon.name>
#          Nicolas Pinto <pinto@alum.mit.edu>
#          Charles Cadieu <cadieu@mit.edu>
#
# License: MIT

# TODO:
# * solver='numpy' or 'scipy' or 'arpack' (faster when n_components is small)
#   (from numpy.linalg import eigh)
#   (from scipy.sparse.linalg import eigsh)


import numpy as np
from numpy import newaxis
from scipy.linalg import eigh

EPSILON = 1e-9
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


def kanalysis(X, Y_true, n_components='all', quantiles=DEFAULT_QUANTILES):
    """Kernel Analysis of a representation `X` w.r.t. target values `Y_true`

    Parameters
    ----------
    X: array-like, shape=[n_samples, n_features]
        Input matrix (representation).

    Y_true: array-like, shape=[n_samples, n_targets]
        Target 'true' matrix (labels).

    n_components: int or 'all', optional
        Number of kPCA components to consider (must be <= n_samples). If
        'all' then consider all components (i.e. n_components=n_samples).

    quantiles: list of floats in ]0.0, 1.0[, optional
        A vector of quantiles of the distribution of pairwise distances
        to build Gaussian kernels.

    Returns
    -------
    ka: array-like, shape=[min(n_components, n_samples) + 1]
        A vector representing the prediction error as a function of the
        number of observed kPCA components.
    """

    assert X.ndim == 2
    assert np.isfinite(X).all()

    assert Y_true.ndim <= 2
    assert np.isfinite(Y_true).all()

    assert len(quantiles) > 0
    quantiles = np.array(quantiles)
    assert quantiles.ndim == 1
    assert (0 < quantiles).all() and (quantiles < 1).all()

    if n_components == 'all':
        n_components = len(X)

    assert (0 < n_components <= len(X))

    # ------------------------------------------------------------------------
    # -- Prepare the data
    # ------------------------------------------------------------------------
    # Compute pairwise squared euclidean (l2) distances
    X_squared = (X ** 2.)
    X_ssq = X_squared.sum(axis=1)[:, newaxis]  # sum-of-squares
    l2_squared = X_ssq + X_ssq.T - 2. * np.dot(X, X.T)

    # Sort them
    l2_squared_sorted = l2_squared.ravel()
    np.sort(l2_squared_sorted)

    # ------------------------------------------------------------------------
    # -- Compute Kernel Analysis for each quantile
    # ------------------------------------------------------------------------
    curves = []
    for quantile in quantiles:

        # Compute generalized gaussian kernel of the form
        # K = exp(-gamma * (dist**2.))
        quantile_idx = int(quantile * len(l2_squared_sorted))
        gamma = 1. / l2_squared_sorted[quantile_idx]
        K = np.exp(-gamma * l2_squared)

        Z = kanalysis_K(K, Y_true, n_components=n_components)

        curves += [Z]

    # ------------------------------------------------------------------------
    # -- Return the minimum values for each kPCA components
    # ------------------------------------------------------------------------
    curves = np.array(curves)
    ka = curves.min(axis=0)

    assert np.isfinite(ka).all()
    return ka


def kanalysis_K(K, Y_true, n_components='all'):
    """Kernel Analysis of `K` w.r.t. target values `Y_true`

    Parameters
    ----------
    K: array-like, shape=[n_samples, n_samples]
        Kernel matrix (RKHS).

    Y_true: array-like, shape=[n_samples, n_targets]
        Target 'true' matrix (labels).

    n_components: int or 'all', optional
        Number of kPCA components to consider (must be <= n_samples). If
        'all' then consider all components (i.e. n_components=n_samples).

    Returns
    -------
    ka: array-like, shape=[min(n_components, n_samples) + 1]
        A vector representing the prediction error as a function of the
        number of observed kPCA components.
    """

    assert K.ndim == 2
    assert K.shape[0] == K.shape[1]
    assert np.isfinite(K).all()
    K = K.copy()

    assert Y_true.ndim <= 2
    assert np.isfinite(Y_true).all()
    Y_true = Y_true.copy()
    if Y_true.ndim == 1:
        Y_true = Y_true[:, newaxis]

    if n_components == 'all':
        n_components = len(K)

    assert (0 < n_components <= len(K))

    # ------------------------------------------------------------------------
    # -- Prepare the data
    # ------------------------------------------------------------------------
    # Center the target values
    Y_center = Y_true - Y_true.mean(axis=0)
    Y_std = Y_center.std()
    if Y_std > EPSILON:
        Y_center = Y_center / Y_std

    # Center the kernel
    K = K.copy()
    K -= K.mean(axis=0)[newaxis, :]
    K -= K.mean(axis=1)[:, newaxis]
    K += K.mean()

    # ------------------------------------------------------------------------
    # -- Compute kPCA components
    # ------------------------------------------------------------------------
    lo_hi = (len(K) - n_components, len(K) - 1)
    eigvals, eigvectors = eigh(K, eigvals=lo_hi)
    assert len(eigvals) == n_components
    assert np.isfinite(eigvectors).all()
    eigvals_idx = (-np.abs(eigvals)).argsort()
    eigvectors = eigvectors[:, eigvals_idx]

    # ------------------------------------------------------------------------
    # -- Projection of labels on the leading kPCA components
    # ------------------------------------------------------------------------
    # (Original formulation using a loop)
    #for v in eigvectors.T:
        #Y_pred += np.dot(np.outer(v, v), Y_center)
        #assert np.isfinite(Y_pred).all()
        #error = ((Y_center - Y_pred) ** 2.).mean()
        #ka = np.concatenate((ka, [error]))

    # (New formulation using tensor outer cumulative product)
    # -- Project full eigenspace onto the target values
    eigvT = eigvectors.T
    eigvTY = np.dot(eigvT, Y_center)

    # -- Cummulative tensordot
    # Outer tensor product
    Y_pred = eigvT[:, :, newaxis] * eigvTY[:, newaxis]
    # Cummulative summation
    Y_pred = Y_pred.cumsum(axis=0)

    # -- Invididual errors
    errors = (Y_pred - Y_center) ** 2.

    # -- Average errors == kernel analysis curve
    ka = errors.reshape(len(errors), -1).mean(-1)

    # -- Full kernel analysis curve
    ka0 = (Y_center ** 2.).mean()
    ka = np.concatenate(([ka0], ka))

    assert np.isfinite(ka).all()
    return ka
