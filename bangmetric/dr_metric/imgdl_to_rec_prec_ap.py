#!/usr/bin/env python

from imgdl_to_cftpfpfn import imgdl_to_cftpfpfn
from cftpfpfn_to_rec_prec import cftpfpfn_to_rec_prec
from rec_prec_to_ap import average_precision
from pythor3.wildwest.dr.utils import imgdl_intersect


def imgdl_to_rec_prec_ap(gv_imgdl, gt_imgdl,
                         min_overlap=0.2,
                         assignment_algorithm='greedy',
                         integration_method='trapz',
                         force_intersection=False,
                         check_sha1=True):

    """
    computes the recall-precision values for every pair of (GT, GV)
    for a given Ground Truth image dictionnary list (imgdl) and a
    list of Given imgdl.

    XXX: update docstring below

    Parameters
    ----------
    gt_imgdl: list
        list of Ground Truth image dictionnaries.

    min_overlap: float (default 0.2)
        overlap above which two overlapping bounding boxes are
        considered a matching pair.

    assignment_algorithm: string (default 'greedy')
        basic algorithm used to build the list of matching (GT, GV)
        pair of bounding boxes in a given image.

    integration_method: string (default 'voc2007')
        integration method used for computing the Average Precision
        from the recall-precision values.
        Possible values are 'voc2007', 'voc2010', 'trapz'.

    Returns
    -------
    a matplotlib (X, Y) plot of recall vs precision for every pair
    (gt_imgdl, gv_imgdl).
    prints the values of the Average Precisions for each pair
    (gt_imgdl, gv_imgdl).

    Warning
    -------
    This program **assumes** that the Ground Truth imgdl does **not**
    contain any 'DCR' bounding box or equivalently that all the Given
    imgdl objects have been cleared of their bounding boxes that overlap
    with any of the Ground Truth 'DCR' regions.
    """

    # first argument must be the Ground Truth imgdl object and it must
    # no be empty
    assert len(gv_imgdl) > 0
    assert len(gt_imgdl) > 0

    # some sanity checks
    assert assignment_algorithm in ['greedy']
    assert integration_method in ['voc2007', 'voc2010', 'trapz']
    assert 0. < min_overlap <= 1.

    # --
    if force_intersection:
        # returns updated imgdl objects with only images from the
        # intersection of the two original imgdl objects.
        gt_imgdl, gv_imgdl = imgdl_intersect(gt_imgdl, gv_imgdl, key='sha1')

    # first we compute the confidences, tp, fp and fn
    cf, tp, fp, fn = imgdl_to_cftpfpfn(
        gv_imgdl, gt_imgdl, min_overlap=min_overlap,
        assignment_algorithm=assignment_algorithm, check_sha1=check_sha1)

    # second we compute the recall and precision arrays from
    # these
    rec, prec = cftpfpfn_to_rec_prec(cf, tp, fp, fn)

    # computing the Average Precision
    ap = average_precision(rec, prec, method=integration_method)

    return rec, prec, ap
