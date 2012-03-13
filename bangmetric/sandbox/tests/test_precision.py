#!/usr/bin/env python

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises
from pythor3.readout.metric import precision_from_boolean as precision

class test_precision(object):
    """
    simple tests for the ``precision_from_boolean`` function
    """
    def test_length(self):
        preds = np.array([False, True, True])
        gt = np.array([True, False])
        assert_raises(AssertionError, precision, preds, gt)

    def test_pass(self):
        preds = np.array([True, False, True, True, False])
        gt = np.array([True, True, False, True, False])
        prec = precision(preds, gt)
        reference = 2. / 3.
        assert_almost_equal(prec, reference, decimal=6)

    def test_pass2(self):
        preds = np.array([True, True, True, True, True])
        gt = np.array([True, True, True, True, True])
        prec = precision(preds, gt)
        reference = 1.
        assert_almost_equal(prec, reference, decimal=6)

    def test_pass3(self):
        preds = np.array([False, False, False, False, False])
        gt = np.array([True, True, True, True, True])
        assert_raises(ZeroDivisionError, precision, preds, gt)

    def test_pass4(self):
        gt = np.array([False, False, False, False, False])
        preds = np.array([True, True, True, True, True])
        prec = precision(preds, gt)
        reference = 0.
        assert_almost_equal(prec, reference, decimal=6)
