#!/usr/bin/env python

"""
test the ``greedy_assignment`` function in ``imgdl_to_rec_prec.py``
"""

from ..imgdl_to_cftpfpfn import greedy_assignment
import numpy as np


def test_greedy_min_overlap_0_5():

    ov = np.array([[0.989,  0.532,  0.916,  0.557, 0.729],
                   [0.495,  0.016,  0.953,  0.807, 0.088],
                   [0.196,  0.665,  0.524,  0.795, 0.746],
                   [0.249,  0.209,  0.877,  0.681, 0.622]])
    confidences = np.array([0.9, 0.81, 0.78, 0.23])

    tp, fp, fn, jmax = greedy_assignment(ov, confidences, 0.5)

    tp_ref, fp_ref, fn_ref, jmax_ref = np.array([1, 1, 1, 0]), \
                                       np.array([0, 0, 0, 1]), \
                                       np.array([0, 1, 0, 0, 1]), \
                                       np.array([0, 2, 3, -1])

    assert (tp == tp_ref).all()
    assert (fp == fp_ref).all()
    assert (fn == fn_ref).all()
    assert (jmax == jmax_ref).all()


def test_greedy_min_overlap_0_9():

    ov = np.array([[0.989,  0.532,  0.916,  0.557, 0.729],
                   [0.495,  0.016,  0.953,  0.807, 0.088],
                   [0.196,  0.665,  0.524,  0.795, 0.746],
                   [0.249,  0.209,  0.877,  0.681, 0.622]])
    confidences = np.array([0.9, 0.81, 0.78, 0.23])

    tp, fp, fn, jmax = greedy_assignment(ov, confidences, 0.9)

    tp_ref, fp_ref, fn_ref, jmax_ref = np.array([1, 1, 0, 0]), \
                                       np.array([0, 0, 1, 1]), \
                                       np.array([0, 1, 0, 1, 1]), \
                                       np.array([0, 2, -1, -1])

    assert (tp == tp_ref).all()
    assert (fp == fp_ref).all()
    assert (fn == fn_ref).all()
    assert (jmax == jmax_ref).all()


def test_greedy_min_overlap_0_45_shuffled_confidences():

    ov = np.array([[0.989,  0.532,  0.916,  0.557, 0.729],
                   [0.995,  0.016,  0.453,  0.807, 0.088],
                   [0.196,  0.665,  0.524,  0.795, 0.746],
                   [0.249,  0.209,  0.877,  0.681, 0.622]])
    confidences = np.array([0.8, 0.9, 0.78, 0.23])

    tp, fp, fn, jmax = greedy_assignment(ov, confidences, 0.45)

    tp_ref, fp_ref, fn_ref, jmax_ref = np.array([0, 1, 1, 1]), \
                                       np.array([1, 0, 0, 0]), \
                                       np.array([0, 1, 0, 0, 1]), \
                                       np.array([-1, 0, 3, 2])

    assert (tp == tp_ref).all()
    assert (fp == fp_ref).all()
    assert (fn == fn_ref).all()
    assert (jmax == jmax_ref).all()
