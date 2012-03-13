"""Computes the Average Precision (AP) from the recall and precision arrays."""

import numpy as np


def average_precision(rec, prec, method='trapz'):
    """Computes the Average Precision (AP) from the recall and precision
    arrays. Different 'integration' methods can be used.

    Parameters
    ----------
    rec: 1D array of floats
        recall values

    prec: 1D array of floats
        precision values

    method: string
        type of 'integration' method used to compute the AP.

    Returns
    -------
    ap: float
        Average Precision

    Notes
    -----
   Code converted from:
    * VOC2007 devkit:
    https://github.com/npinto/VOCdevkit/blob/VOC2007/VOCcode/VOCevalcls.m#L31

    * VOC2010 devkit:
    https://github.com/npinto/VOCdevkit/blob/VOC2010/VOCcode/VOCevalcls.m#L34
    https://github.com/npinto/VOCdevkit/blob/VOC2010/VOCcode/VOCap.m
    """

    assert rec.ndim == 1
    assert prec.ndim == 1

    if rec.size == 0:
        return 0

    assert rec.size > 0
    assert prec.size > 0
    assert rec.size == prec.size

    assert method in ['voc2007', 'voc2010', 'trapz']

    if method == 'voc2007':
        ap = 0.
        rng = np.arange(0, 1.1, .1)
        for th in rng:
            p = prec[rec >= th]
            if len(p) > 0:
                ap += p.max() / rng.size

    elif method == 'voc2010':
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        rng = np.arange(len(mpre) - 1)[::-1]
        for i in rng:
            mpre[i] = max(mpre[i], mpre[i+1])
        sel = np.nonzero(mrec[1:] != mrec[0:-1])[0] + 1
        ap = ((mrec[sel] - mrec[sel-1]) * mpre[sel]).sum()
        if np.isnan(ap):
            ap = 0.

    elif method == 'trapz':

        from scipy.integrate import trapz

        if rec[0] != 0.:
            rec = np.concatenate(([0.], rec))
            prec = np.concatenate(([prec[0]], prec))

        ap = trapz(prec, rec)

    return ap
