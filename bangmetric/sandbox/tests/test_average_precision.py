#!/usr/bin/env python

import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises
from pythor3.readout.metric import average_precision

class test_average_precision(object):
    """
    simple tests for the ``average_precision`` function
    """
    def test_length(self):
        """
        should fail because preds and gt don't have the same
        dimension
        """
        preds = np.array([0.1, 0.2, 0.3])
        gt = np.array([1., -1.])
        assert_raises(AssertionError, average_precision, preds, gt)

    def test_fail1(self):
        """
        should fail because "dummy" is not a supported method
        of integration
        """
        preds = np.array([0.87, 0.21, 0.35])
        gt = np.array([1., -1., 1.])
        assert_raises(ValueError, average_precision, preds, gt,
                integration_type='dummy')

    def test_fail2(self):
        """
        should fail because one of the prediction is negative
        """
        preds = np.array([0.87, -0.21, 0.35])
        gt = np.array([1., -1., 1.])
        assert_raises(AssertionError, average_precision, preds, gt)

    def test_fail3(self):
        """
        should fail because one of the prediction is negative, the
        name of the integration method being irrelevant
        """
        preds = np.array([0.87, -0.21, 0.35])
        gt = np.array([1., -1., 1.])
        assert_raises(AssertionError, average_precision, preds, gt,
                integration_type='dummy')

    def test_pass1(self):
        """
        should pass because the predictions exactly match the
        ground truth
        """
        preds = np.array([1., 1., 1.])
        gt = np.array([1., 1., 1.])
        ap = average_precision(preds, gt)
        reference = 1.
        assert_almost_equal(ap, reference, decimal=6)

    def test_pass2(self):
        """
        should pass because the predictions exactly match the
        ground truth and that the integration method is irrelevant
        """
        preds = np.array([1., 1., 1.])
        gt = np.array([1., 1., 1.])
        ap = average_precision(preds, gt, integration_type='voc2007')
        reference = 1.
        assert_almost_equal(ap, reference, decimal=6)

    def test_pass3(self):
        """
        should pass because the predictions are all wrong
        """
        preds = np.array([1., 1., 1.])
        gt = np.array([-1., -1., -1.])
        ap = average_precision(preds, gt)
        reference = 0.
        assert_almost_equal(ap, reference, decimal=6)

    def test_pass4(self):
        """
        should pass because the predictions are all wrong and that
        the integration method is irrelevant
        """
        preds = np.array([1., 1., 1.])
        gt = np.array([-1., -1., -1.])
        ap = average_precision(preds, gt, integration_type='voc2007')
        reference = 0.
        assert_almost_equal(ap, reference, decimal=6)

    def test_pass5(self):
        """
        should pass because the true average precision is known for
        this example
        """
        pass
