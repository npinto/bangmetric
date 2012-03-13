import numpy as np

# TODO: consider using scikits.learn.metrics here
# see http://scikit-learn.sourceforge.net/modules/classes.html#module-scikits.learn.metrics
# e.g. accuracy = scikits.learn.metrics.zero_one_score

def accuracy(gv, gt):
    """XXX: docstring for accuracy"""
    assert len(gv) == len(gt)
    accuracy = (gv == gt).mean()
    return accuracy


def dprime(gv, gt):
    """XXX: docstring for dprime"""
    idx_pos = gt > 0
    idx_neg = -idx_pos
    gt_pos_mean = gv[idx_pos].mean()
    gt_neg_mean = gv[idx_neg].mean()
    gt_pos_var = gv[idx_pos].var(ddof=1)
    gt_neg_var = gv[idx_neg].var(ddof=1)
    dprime = (
        (gt_pos_mean - gt_neg_mean)
        /
        np.sqrt((gt_pos_var + gt_neg_var) / 2)
    )
    return dprime


def average_precision(gv, gt, method='voc2010'):
    """XXX: docstring for average_precision

    TODO: tests vs. VOCdevkit

    Notes:
    ------

    Code converted from:
    * VOC2007 devkit:
    https://github.com/npinto/VOCdevkit/blob/VOC2007/VOCcode/VOCevalcls.m#L31

    * VOC2010 devkit:
    https://github.com/npinto/VOCdevkit/blob/VOC2010/VOCcode/VOCevalcls.m#L34
    https://github.com/npinto/VOCdevkit/blob/VOC2010/VOCcode/VOCap.m
    """

    assert method in ('voc2007', 'voc2010')
    assert len(gv) == len(gt)

    si = np.argsort(-gv)
    tp = np.cumsum(np.single(gt[si] > 0))
    fp = np.cumsum(np.single(gt[si] < 0))
    rec = tp / np.sum(gt > 0)
    prec = tp / (fp + tp)

    if method == 'voc2007':
        ap = 0
        rng = np.arange(0, 1.1, .1)
        for th in rng:
            p = prec[rec >= th]
            if len(p) > 0:
                ap += p.max() / rng.size

    elif method == 'voc2010':
        mrec = np.concatenate(([0], rec, [1]))
        mpre = np.concatenate(([0], prec, [0]))
        rng = np.arange(len(mpre) - 1)[::-1]
        for i in rng:
            mpre[i] = max(mpre[i], mpre[i+1])
        sel = np.nonzero(mrec[1:] != mrec[0:-1])[0] + 1
        ap = ((mrec[sel] - mrec[sel-1]) * mpre[sel]).sum()

    assert 0 <= ap <= 1
    return ap
