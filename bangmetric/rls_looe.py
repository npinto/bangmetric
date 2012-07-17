"""Leave-One-Out Cross-Validation Stability using Regularized Least-Squares

See `rls_looe` documentation for more information.

Status: experimental.
"""


# Authors: Nicolas Pinto <pinto@alum.mit.edu>
#
# License: BSD 3-clause


import numpy as np

# the following import is temporary while waiting for sklearn-0.12
# see sklearn's PR #958:
# https://github.com/scikit-learn/scikit-learn/pull/958
from _sklearn_ridge import RidgeCV


def rls_looe(X, Y,
               l2_regularizations=[
                   1e6, 1e5, 1e4, 1e3, 1e3, 1e1,
                   1e0,
                   1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,
               ],
               return_cv_values=False,
              ):
    """Regularized Least-Squares (aka Ridge Regression) Leave-One-Out Error.

    Inspired from the leave-one-out cross-validation (CV_loo) stability
    proposed in Mukherjee et al. (2003) as a statistical form of
    stability.  In their paper, they showed that CV_loo is (1)
    sufficient for generalization, (2) necessary and sufficient for
    consistency when using bounded losses.

    This function returns the mininum leave-one-out error (scalar) over
    the `l2_regularizations` values provided, unless
    `return_cv_values=True` in which case all cross-validaton values are
    returned (tensor).


    Parameters
    ----------

    X : array-like, shape = [n_samples, n_features]
        Input array.

    Y : array-like, shape = [n_samples] or [n_samples, n_responses]
        Target array.

    return_cv_values : boolean, optional, default=False
        Flag to return all the cross-validation values `cv_values` in
        addition to leave-one-out errors `looe`.


    Returns
    -------

    looe : float
        Minimum leave-one-out error over the `l2_regularizations`.

    cv_values : array-like, \
                shape = [n_samples, n_responses, n_l2_regularizations], \
                optional


    References
    ----------

    * Statistical Learning: CV_loo stability is sufficient for
    generalization and necessary and sufficient for consistency of
    Empirical Risk Minimization (2003)
    Sayan Mukherjee, Partha Niyogi, Tomaso Poggio and Ryan Rifkin
    http://cbcl.mit.edu/publications/ps/wellposedness-consistency.pdf

    * Notes on Regularized Least SquareNotes on Regularized Least
    Squares (2007)
    Ryan M. Rifkin and Ross A. Lippert
    http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf
    """

    assert type(X) == np.ndarray
    assert X.ndim == 2

    assert type(Y) == np.ndarray
    assert Y.ndim <= 2

    if Y.ndim == 1:
        Y = Y[:, np.newaxis]

    l2_regularizations = np.unique(l2_regularizations)
    l2_regularizations = \
            l2_regularizations[(-l2_regularizations).argsort()]

    # -- fit the data and get back the values of each loo
    # (for each regularization parameter)
    # Both RidgeCV and Mukherjee et al. use the square loss (see Section 2.2)
    ridge = RidgeCV(alphas=l2_regularizations, store_cv_values=True)
    ridge.fit(X, Y)

    cv_values = ridge.cv_values_
    assert np.isfinite(cv_values).all()

    # -- compute minimum loo error
    looe = cv_values.reshape(-1, len(l2_regularizations))
    looe = looe.mean(axis=0).min()

    if return_cv_values:
        return looe, cv_values
    else:
        return looe
