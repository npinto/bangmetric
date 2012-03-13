"""Test suite for the ``precision_recall`` module"""

import numpy as np
from pytest import raises
from bangmetric import average_precision

# XXX: test recall()
# XXX: test precision()

def test_length():
    # should fail because y_pred and y_true don't have the same dimension
    y_pred = np.array([0.1, 0.2, 0.3])
    y_true = np.array([1., -1.])
    raises(AssertionError,
           average_precision, y_true, y_pred)

def test_fail1():
    # should fail because "dummy" is not a supported method
    # of integration
    y_pred = np.array([0.87, 0.21, 0.35])
    y_true = np.array([1., -1., 1.])
    raises(AssertionError,
           average_precision, y_pred, y_true, integration='dummy')

#def test_pass1():
    ## should pass because the predictions exactly match the
    ## ground truth
    #y_pred = np.array([1., 1., 1.])
    #y_true = np.array([1., 1., 1.])
    #ap = average_precision(y_pred, y_true)
    #reference = 1.
    #assert_almost_equal(ap, reference, decimal=6)

#def test_pass2():
    ## should pass because the predictions exactly match the ground truth
    ## and that the integration method is irrelevant
    #y_pred = np.array([1., 1., 1.])
    #y_true = np.array([1., 1., 1.])
    #ap = average_precision(y_pred, y_true, integration_type='voc2007')
    #reference = 1.
    #assert_almost_equal(ap, reference, decimal=6)

#def test_pass3():
    ## should pass because the predictions are all wrong
    #y_pred = np.array([1., 1., 1.])
    #y_true = np.array([-1., -1., -1.])
    #ap = average_precision(y_pred, y_true)
    #reference = 0.
    #assert_almost_equal(ap, reference, decimal=6)

#def test_pass4():
    ## should pass because the predictions are all wrong and that the
    ## integration method is irrelevant
    #y_pred = np.array([1., 1., 1.])
    #y_true = np.array([-1., -1., -1.])
    #ap = average_precision(y_pred, y_true, integration_type='voc2007')
    #reference = 0.
    #assert_almost_equal(ap, reference, decimal=6)
