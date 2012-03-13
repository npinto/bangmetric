#!/usr/bin/env python

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises
from pythor3.readout.metric import accuracy_from_boolean as accuracy

class test_accuracy(object):
    """
    simple tests for the ``accuracy_from_boolean`` function
    """
    def test_length(self):
        preds = np.array([False, True, True])
        gt = np.array([True, False])
        assert_raises(AssertionError, accuracy, preds, gt)

    def test_pass(self):
        preds = np.array([True, False, True, True, False])
        gt = np.array([True, True, False, True, False])
        acc = accuracy(preds, gt)
        reference = 3. / 5.
        assert_almost_equal(acc, reference, decimal=6)

    def test_pass2(self):
        preds = np.array([True, True, True, True, True])
        gt = np.array([True, True, True, True, True])
        acc = accuracy(preds, gt)
        reference = 1.
        assert_almost_equal(acc, reference, decimal=6)

    def test_pass3(self):
        preds = np.array([False, False, False, False, False])
        gt = np.array([True, True, True, True, True])
        acc = accuracy(preds, gt)
        reference = 0.
        assert_almost_equal(acc, reference, decimal=6)
