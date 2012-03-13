#!/usr/bin/env python

"""
Format for an imgdl objects
===========================

An ``imgdl`` object is a list of dictionnaries. There is **one**
dictionnary per image in the set. We call such a dictionnary an
``imgd`` object. So :

        >>> len(imgdl)
        >>> number_of_images_in_set

Format of an imgd object
========================

An ``imgd`` object is a dictionnary. It contains all the necessary
information concerning a given image in the set.

We require the existence of at least **3** keys:

1. a *filename* key that gives the full path to the original image
   file on disk
2. a *sha1* key that gives the unique sha1_ signature of the original
   image file
3. an *objects* key which designates the list of bounding box objects
   found in the image. The corresponding value is an ``objdl`` object
   (see next paragraph).

Here is a typical example of how an ``imgd`` dictionnary looks like :

        >>> imgd
        >>> {'filename': '/home/poilvert/.scikit-data/pascal/VOC2007/' + \
                         'VOCdevkit/VOC2007/JPEGImages/003318.jpg',
             'objects': objdl,
             'sha1': '2a3432a2412f1269c8e7c33fa6053454634ab908'}

Format for an objdl object
==========================

An ``objdl`` object is a list of dictionnaries. There is **one**
dictionnary per "object" (generally a bounding box) found in the image.
We call such a dictionnary an ``objd`` object. So :

        >>> len(objdl)
        >>> number_of_objects_in_image

Format of a objd object
=======================

An ``objd`` object is a dictionnary. It contains all the necessary
information concerning a single object found in the image (generally
a bounding box object).

We require the existence of at least **3** keys for a bounding box object:

1. a *name* key that designates the object class name (e.g. 'Car',
   'aeroplane', etc...)
2. a *confidence* key, which gives the confidence with which the system
   (or model) delivered that bounding box. For *Ground Truth* bounding
   boxes, this confidence is generally set to 1.
3. a *bounding_box* key for which the associated value designates the
   geometry of the bounding box.

For a DARPA bounding box object, we have the following structure for a
bounding box dictionnary :

        >>> DARPA_objd["bounding_box"]
        >>> {"x1": 237, "y1": 100, "x2": 287, "y2": 100, "x3": 287,
             "y3": 200, "x4": 237, "y4": 200}

For a Pascal VOC bounding box object, we have the following structure
for a bounding box dictionnary :

        >>> Pascal_VOC_objd["bounding_box"]
        >>> {"x_min": 237, "y_min": 100, "x_max": 287, "y_max": 200}

.. external links

.. _sha1: http://en.wikipedia.org/wiki/SHA-1
"""

import numpy as np
from objdl_to_overlap import (gvgt_overlap, objdl_to_xyminmax)


def imgdl_to_cftpfpfn(gv_imgdl, gt_imgdl, min_overlap=0.2,
                      assignment_algorithm='greedy',
                      check_sha1=True,
                     ):
    """Takes a list of Given (GV) image dictionnaries and a list of
    Ground Truth (GT) dictionnaries and returns the recall and precision
    arrays.

    Parameters
    ----------
    gv_imgdl: list of dictionnaries
        The format for an ``imgd`` object is given in the introductory
        docstrings. list of Given image dictionnaries.

    gt_imgdl: list of dictionnaries
        List of Ground Truth image dictionnaries.

    min_overlap: float
        threshold above which two overlapping bounding boxes are
        considered a matching pair. The overlap needs to be *superior* or
        *equal* to ``min_overlap`` in order to be considered a match.

    assignment_algorithm: string
        designates the type of algorithm used when building the (GV, GT)
        matching pairs of bboxes.  Default: greedy (hungarian is also
        possible. Note in that latter case that the basic assignment
        problem scales in N**3 where N is the minimum between the number
        of GV bboxes and GT bboxes)

    check_sha1: bool
        force check on image sha1 signatures for consistency

    Returns
    -------
    cf: list of 1D arrays of floats
        confidence values for every GV bbox in each image.

    tp: list of 1D array of integers
        Values for the True Positives in the order of the Given bounding
        boxes in each image. This means that these tp values are not
        sorted by GV bbox confidences.

    fp: list of 1D array of integers
        Values for the False Positives in the order of the Given bounding
        boxes in each image. This means that these fp values are not
        sorted by GV bbox confidences.

    fn: 1D array of integers
        Values for the False Negatives (i.e. the missed detects or GT bboxes
        not assigned to any GV bbox) in the order of the Ground Truth
        bounding boxes.

    Notes
    -----
    The function asserts that:
        - the input lists should have equal lengths
        - the lists should contain **exactly** the same images
        - the ``min_overlap`` parameter **cannot** be strictly zero
        - the image dictionnaries should all contain at least a key
          called ``objects``
    """

    # some consistency checks
    msg = "not the same image number in Ground Truth and Givens"
    assert len(gv_imgdl) == len(gt_imgdl), msg

    msg = "Ground Truth and Givens are empty"
    assert len(gv_imgdl) > 0, msg

    if check_sha1:
        msg = "Ground Truth and Givens do not have the same image sequence"
        gv_sha1s = [imgd['sha1'] for imgd in gv_imgdl]
        gt_sha1s = [imgd['sha1'] for imgd in gt_imgdl]
        assert gv_sha1s == gt_sha1s, msg
        msg = "some images have been duplicated"
        assert len(np.unique(gv_sha1s)) == len(gv_sha1s), msg

    msg = "some image dictionnaries have a missing ``objects`` attribute"
    gv_objects = np.array(['objects' in imgd for imgd in gv_imgdl])
    gt_objects = np.array(['objects' in imgd for imgd in gt_imgdl])
    assert gv_objects.sum() == gv_objects.size, msg
    assert gt_objects.sum() == gt_objects.size, msg

    msg = "minimum overlap cannot be strictly equal to 0"
    assert float(min_overlap) > 0., msg

    assert assignment_algorithm.lower() in ['greedy', 'hungarian']

    # we extract the GV confidences on a per image basis
    confidences = []
    for gv_imgd in gv_imgdl:
        objdl = gv_imgd['objects']
        conf = [objd['confidence'] for objd in objdl]
        confidences.append(np.array(conf))

    # loop over the GV and GT images and compute tp, fp, fn
    # for each of these images. We store the resuling arrays
    # in three lists
    tp, fp, fn = [], [], []

    # we will also store for each image, the position indices
    # of all the matching GT bboxes. If a GV bbox does not have
    # an assigned GT bbox, then the position index is defined to
    # be -1
    gt_matching_idx = []

    for img_idx, (gv_imgd, gt_imgd) in enumerate(zip(gv_imgdl, gt_imgdl)):

        # we extract the Given and Ground Truth bboxes
        gv_objdl, gt_objdl = gv_imgd['objects'], gt_imgd['objects']
        gv_objdl_len, gt_objdl_len = len(gv_objdl), len(gt_objdl)

        # then we solve for the "corner cases"
        if gv_objdl_len == 0 and gt_objdl_len == 0:
            tp.append(np.array([], dtype=int))
            fp.append(np.array([], dtype=int))
            fn.append(np.array([], dtype=int))
            gt_matching_idx.append(np.array([], dtype=int))

        elif gv_objdl_len == 0 and gt_objdl_len > 0:
            tp.append(np.array([], dtype=int))
            fp.append(np.array([], dtype=int))
            fn.append(np.ones(gt_objdl_len, dtype=int))
            gt_matching_idx.append(np.array([], dtype=int))

        elif gv_objdl_len > 0 and gt_objdl_len == 0:
            tp.append(np.zeros(gv_objdl_len, dtype=int))
            fp.append(np.ones(gv_objdl_len, dtype=int))
            fn.append(np.array([], dtype=int))
            gt_matching_idx.append(-np.ones(gv_objdl_len, dtype=int))

        # now the most interesting case where we need to assign a GT
        # bbox to every GV (if possible)
        else:
            # building the matrix of overlap
            overlap_matrix, _, _ = gvgt_overlap(
                *
                tuple(objdl_to_xyminmax(gv_objdl))
                +
                tuple(objdl_to_xyminmax(gt_objdl))
                )

            # computing the tp, fp and fn arrays for that image along
            # with the array of GT matching position indices
            if assignment_algorithm.lower() == 'greedy':

                img_gv_conf = confidences[img_idx]
                img_tp, img_fp, img_fn, img_jmax = \
                        greedy_assignment(overlap_matrix,
                                          img_gv_conf,
                                          min_overlap)

                tp.append(img_tp)
                fp.append(img_fp)
                fn.append(img_fn)
                gt_matching_idx.append(img_jmax)

            elif assignment_algorithm.lower() == 'hungarian':

                img_tp, img_fp, img_fn, img_jmax = \
                        hungarian_assignment(overlap_matrix,
                                             min_overlap)

                tp.append(img_tp)
                fp.append(img_fp)
                fn.append(img_fn)
                gt_matching_idx.append(img_jmax)

    return np.concatenate(confidences), np.concatenate(tp), \
           np.concatenate(fp), np.concatenate(fn)


def hungarian_assignment(overlap_matrix, min_overlap):
    """TO DO"""
    raise(NotImplementedError)


def greedy_assignment(overlap_matrix, confidences, min_overlap):

    from numpy import ma

    assert overlap_matrix.shape[0] > 0
    assert overlap_matrix.shape[1] > 0

    assert confidences.size == overlap_matrix.shape[0]

    # defining some useful parameters
    N = overlap_matrix.shape[0]
    M = overlap_matrix.shape[1]

    ##
    # STEP 1 : we threshold the overlap matrix
    ##
    ovth = np.where(overlap_matrix < min_overlap, 0., overlap_matrix)

    nonemptyrows = (ovth.sum(1) > 0.)
    nonemptycols = (ovth.sum(0) > 0.)

    mask = np.outer(nonemptyrows, nonemptycols)

    # new set of parameters
    rN = nonemptyrows.sum()
    rM = nonemptycols.sum()

    if nonemptyrows.sum() == 0:
        tp = np.zeros(N, dtype=int)
        fp = np.ones(N, dtype=int)
        fn = np.ones(M, dtype=int)
        gt_argmax = -np.ones(N, dtype=int)
        return tp, fp, fn, gt_argmax
    else:
        # selecting only the non-empty rows and cols from "ovth"
        rovth = ovth[nonemptyrows][:, nonemptycols]

        # also we create a masked array much similar to rovth but
        # the advantage is that the former "remember" the column
        # index of the elements. We will use that array to extract
        # the argmax on each rows
        masked_ovth = ma.masked_array(ovth, mask=-mask)

        # pulling the argmax for every row in masked_ovth
        rgt_argmax = masked_ovth.argmax(1)[nonemptyrows]

        # pulling the same argmax but from matrix rovth
        jmax = rovth.argmax(1)

        assert rgt_argmax.size == rN
        assert rovth.shape[0] == rN
        assert rovth.shape[1] == rM

        # creating a binary matrix which equals 1 at the
        # argmax position for each row in rovth and 0
        # otherwise
        jmax_2Dto1D = jmax + rM * np.arange(rN)

        # ba(i) stands for "binary assignment" for step i
        ba1 = np.zeros((rN, rM), dtype=int)
        ba1.flat[jmax_2Dto1D] = 1

        assert (ba1.sum(1) == 1).all()

        ##
        # STEP 2 : we take care of the potential multiple match
        ##
        rcf = confidences[nonemptyrows]

        # rcfsd stands for "reduced confidences sorted in decreasing order"
        rcfsd = (-rcf).argsort()

        # rev_idx is the inverse mapping of rcfsd
        rev_idx = rcfsd.argsort()

        sba1 = ba1[rcfsd, :]

        # we only select the columns of sba1 that have more than 1 as a
        # sum. This indicates multiple GT assignments.
        sel_cols = (sba1.sum(0) > 1)

        if sel_cols.sum() == 0:
            ba3 = sba1[rev_idx, :]
            assert (ba3 == ba1).all()
        else:
            rsba1 = sba1[:, sel_cols]

            fitness = np.arange(rsba1.shape[0], 0, -1)

            if rsba1.ndim == 1:
                mrsba1 = rsba1 * fitness
            elif rsba1.ndim == 2:
                mrsba1 = rsba1 * fitness[:, np.newaxis]
            else:
                raise(ValueError('wrong dimensionality for "rsba1" array'))

            sba1_newcols = (mrsba1 == mrsba1.max(0)).astype(int)

            sba1[:, sel_cols] = sba1_newcols

            ba3 = sba1[rev_idx, :]

            rgt_argmax[(ba3.sum(1) == 0)] = -1

        ##
        # STEP 3 : finally we return the final ba array
        ##
        ba_final = np.zeros((N, M), dtype=int)
        ba_final[mask] = ba3.ravel()

        assert (ba_final.sum(1) <= 1).all()
        assert (ba_final.sum(0) <= 1).all()

        ##
        # STEP 4 : return tp, fp and fn
        ##
        tp = (ba_final.sum(1) == 1).astype(int)
        fp = (ba_final.sum(1) == 0).astype(int)
        fn = (ba_final.sum(0) == 0).astype(int)

        # update the gt_argmax array
        gt_argmax = -np.ones(N, dtype=int)
        gt_argmax[nonemptyrows] = rgt_argmax

        assert (tp + fp).sum() == N
        assert fn.sum() <= M
        assert gt_argmax.size == N
        assert jmax.size == rN

        return tp, fp, fn, gt_argmax

# vim: set ts=4 sw=4 tw=73 :
