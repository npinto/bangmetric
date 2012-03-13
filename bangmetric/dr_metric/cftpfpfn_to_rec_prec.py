#!/usr/bin/env python

"""
This module allows to compute the recall and precision values
from the confidences, false positives and false negatives.
"""


def cftpfpfn_to_rec_prec(cf, tp, fp, fn):
    """
    This function will simply sort the tp and fp arrays according
    to the confidence values. Then we will compute the cumulative
    sums and use the basic definitions for recall and precision.

    Parameters
    ----------
    cf: 1D array of floats
        confidence values

    tp: 1D array of integers
        1 if True Positive, 0 otherwise

    fp: 1D array of integers
        1 if False Positive, 0 otherwise
        (should be -tp if understood as an array of booleans)

    fn: 1D array of integers
        1 if False Negative of Missed Detect, 0 otherwise

    Returns
    -------
    rec: 1D array of floats
        recall values

    prec: 1D array of floats
        precision values
    """

    assert cf.ndim == 1
    assert tp.ndim == 1
    assert fp.ndim == 1
    assert fn.ndim == 1
    assert tp.size == fp.size
    assert (tp+fp).sum() == tp.size

    # sorted indices
    si = (-cf).argsort()

    # re-arranging tp and fp according to the confidences
    stp = tp[si]
    sfp = fp[si]

    # cumulated tp and fp arrays
    ctp = stp.cumsum()
    cfp = sfp.cumsum()
    ngt = fn.size

    # if there is no Ground Truth there's a problem
    msg = "no Ground Truth given so cannot define recall value"
    assert ngt > 0, msg

    # definitions of recall and precision (we use the number
    # of elements in the False Negative array to compute the
    # number of Ground Truth)
    rec = 1. * ctp / ngt
    prec = 1. * ctp / (ctp + cfp)

    return rec, prec
