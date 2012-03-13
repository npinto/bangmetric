import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import numpy as np
from scipy import linalg as la


# ---------------------------------------------------------------------
def idc_rls(X, Y,
            l2_regularizations=[
                1e6, 1e5, 1e4, 1e3, 1e3, 1e1,
                1e0,
                1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,
            ],
    ):

    log.info(">>> RLS...")

    l2_regularizations = np.unique(l2_regularizations)
    l2_regularizations = \
            l2_regularizations[(-l2_regularizations).argsort()]

    # -- SVD
    n, d = X.shape

    U, s, Vh = la.svd(X, full_matrices=False)
    assert U.shape == (n, d)
    assert s.shape == (d,)
    assert Vh.shape == (d, d)

    # -- optimal 'dual_coef_'
    UTxY = np.dot(U.T, Y)

    rmses = []
    for l in l2_regularizations:
        log.info('l2_regularization=%s', l)

        S2l = np.diag(
            ((s ** 2.) + l) ** (-1.)
            -
            l ** (-1.)
        )

        UxS2l = np.dot(U, S2l)

        UxS2lxUTxY = np.dot(UxS2l, UTxY)
        #print UxS2lxUTxY.shape

        lY = (l ** (-1.)) * Y

        dual_coef_ = UxS2lxUTxY + lY
        #print dual_coef_
        #print dual_coef_.shape
        #print dual_coef_[:, 0]

        # -- LOOE
        UxS2lxUTl = np.dot(UxS2l, U.T) + (l ** (-1.))
        diagUxS2lxUTl = np.diag(UxS2lxUTl)[:, np.newaxis]

        looe = dual_coef_ / diagUxS2lxUTl
        #print looe
        #print looe.shape

        # -- RMSE
        #rmse = 1. - np.sqrt((looe ** 2.).mean())
        rmse = np.sqrt((looe ** 2.).mean())
        log.info('rmse=%s', rmse)
        #print rmse
        rmses += [rmse]

        # -- coef_
        #coef_ = np.dot(dual_coef_.T, X)
        #print coef_.shape

    return np.min(rmses)
