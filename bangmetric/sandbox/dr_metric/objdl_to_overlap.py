#!/usr/bin/env python

import numpy as np


def objdl_to_overlap(gv_objdl, gt_objdl):
    args = tuple(objdl_to_xyminmax(gv_objdl))
    args += tuple(objdl_to_xyminmax(gt_objdl))
    return gvgt_overlap(*args)


def gvgt_overlap(gvxmin, gvxmax, gvymin, gvymax,
                 gtxmin, gtxmax, gtymin, gtymax):

    # -- pre-conditions
    assert gvxmin.ndim == 1
    assert gvxmax.ndim == 1
    assert gvymin.ndim == 1
    assert gvymax.ndim == 1

    assert gtxmin.ndim == 1
    assert gtxmax.ndim == 1
    assert gtymin.ndim == 1
    assert gtymax.ndim == 1

    gv_size = gvxmin.size
    gt_size = gtxmin.size

    # -- multi-dimensional mesh grid
    gtxmin, gvxmin = np.meshgrid(gtxmin, gvxmin)
    assert gtxmin.shape == (gv_size, gt_size)
    assert gvxmin.shape == (gv_size, gt_size)

    gtxmax, gvxmax = np.meshgrid(gtxmax, gvxmax)
    assert gtxmax.shape == (gv_size, gt_size)
    assert gvxmax.shape == (gv_size, gt_size)

    gtymin, gvymin = np.meshgrid(gtymin, gvymin)
    assert gtymin.shape == (gv_size, gt_size)
    assert gvymin.shape == (gv_size, gt_size)

    gtymax, gvymax = np.meshgrid(gtymax, gvymax)
    assert gtymax.shape == (gv_size, gt_size)
    assert gvymax.shape == (gv_size, gt_size)

    # -- intersection area (ia)
    interxmin = np.dstack((gvxmin, gtxmin)).max(-1)
    interxmax = np.dstack((gvxmax, gtxmax)).min(-1)
    interymin = np.dstack((gvymin, gtymin)).max(-1)
    interymax = np.dstack((gvymax, gtymax)).min(-1)

    interw = interxmax - interxmin + 1
    interw[interw < 0] = 0
    interh = interymax - interymin + 1
    interh[interh < 0] = 0

    ia = interw * interh
    assert (ia >= 0).all()
    assert ia.shape == (gv_size, gt_size)

    # -- union area (ua)
    gtw = gtxmax - gtxmin + 1
    gth = gtymax - gtymin + 1
    gvw = gvxmax - gvxmin + 1
    gvh = gvymax - gvymin + 1

    ua = (gtw * gth) + (gvw * gvh) - ia
    assert (ua > 0).all()
    assert ua.shape == (gv_size, gt_size)

    # -- overlap
    overlap = 1. * ia / ua

    # -- post-conditions
    assert overlap.shape == (gv_size, gt_size)
    assert np.isfinite(overlap).all()
    assert (0 <= overlap).all() and (overlap <= 1).all()

    return overlap, ia, ua


def objdl_to_xyminmax(objdl):

    assert len(objdl) > 0

    xyminmax = np.array([objd_to_xyminmax(objd) for objd in objdl]).T
    assert xyminmax.shape[0] == 4
    assert xyminmax.shape[1] == len(objdl)

    assert ((xyminmax[1] - xyminmax[0]) >= 0).all()
    assert ((xyminmax[3] - xyminmax[2]) >= 0).all()

    return xyminmax


def objd_to_xyminmax(objd):

    assert 'bounding_box' in objd

    bb = objd['bounding_box']

    x1, x2, x3, x4 = bb['x1'], bb['x2'], bb['x3'], bb['x4']
    y1, y2, y3, y4 = bb['y1'], bb['y2'], bb['y3'], bb['y4']

    xmin = min([x1, x2, x3, x4])
    xmax = max([x1, x2, x3, x4])
    ymin = min([y1, y2, y3, y4])
    ymax = max([y1, y2, y3, y4])

    return xmin, xmax, ymin, ymax
