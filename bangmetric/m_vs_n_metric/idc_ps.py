#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import numpy as np

from spear import spearman


def get_corrcoef_triangular(arr):
    assert arr.ndim == 2
    cc = np.corrcoef(arr)
    yy, xx = np.mgrid[:cc.shape[0], :cc.shape[1]]
    cc = cc[yy > xx].ravel()
    return cc


# ---------------------------------------------------------------------
def idc_ps(X, Y):
    #log.info(">>> Pearson...")
    X = get_corrcoef_triangular(X)
    Y = get_corrcoef_triangular(Y)

    #log.info(">>> Spearman...")
    rho = spearman(X, Y)
    #log.info("spearman=%s", rho)

    return rho
