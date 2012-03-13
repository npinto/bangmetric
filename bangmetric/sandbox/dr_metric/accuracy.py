import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import numpy as np
from pythor3.readout import metric

def balanced_accuracy(gv, gt):
    gv = np.array(gv)
    gt = np.array(gt)
    assert gv.size == gt.size
    gt_pos_idx = gt > 0
    gt_neg_idx = gt <= 0
    acc_pos = (gv[gt_pos_idx] > 0).mean()
    acc_neg = (gv[gt_neg_idx] <= 0).mean()
    log.debug('acc_pos: %.3f' % acc_pos)
    log.debug('acc_neg: %.3f' % acc_neg)
    return 0.5 * acc_pos + 0.5 * acc_neg

def accuracy(gv, gt):
    gv = np.array(gv)
    gt = np.array(gt)
    assert gv.size == gt.size
    return (gv == gt).mean()


average_precision = metric.average_precision


# vim: set ts=4 sw=4 tw=73 :
