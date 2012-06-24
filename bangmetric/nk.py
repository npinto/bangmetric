import numpy as np

from .correlation import spearman


def triangular_corrcoef(arr):
    """Returns the triangular representation of the inner correlation
    coefficients of `X`
    """
    assert arr.ndim == 2
    cc = np.corrcoef(arr)
    yy, xx = np.mgrid[:cc.shape[0], :cc.shape[1]]
    cc = cc[yy > xx].ravel()
    return cc


# ---------------------------------------------------------------------
def nk_similarity(A, B):
    """Computes Nikolaus Kriegeskorte's similarity metric on two
    representations `A` and `B`.

    Parameters
    ----------
    A: array-like, shape = [n_samples, n_featuresA]

    B: array-like, shape = [n_samples, n_featuresB]


    Returns
    -------
    rho: float
        Nikolaus Kriegeskorte's population similarity metric.


    References
    ----------
    * http://www.mrc-cbu.cam.ac.uk/people/nikolaus.kriegeskorte/hmit.html
    """

    A = triangular_corrcoef(A)
    B = triangular_corrcoef(B)

    rho = spearman(A, B)

    return rho
