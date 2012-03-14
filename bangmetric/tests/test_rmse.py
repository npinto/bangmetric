"""Test suite for the ``rmse`` module"""

import numpy as np
from pytest import raises

from bangmetric import rmse

ATOL = 1e-6


def test_error_length():
    y_pred = np.array([False, True, True])
    y_true = np.array([True, False])
    raises(AssertionError, rmse, y_pred, y_true)


def test_basic():
    y_true = np.array([False, True, True, True, False, False, False, True])
    y_pred = np.array([0.491, -0.1, 0.64, 1.52, -0.23, -0.23, 1.579, 0.76])

    err = rmse(y_true, y_pred)
    reference = 0.75064322417510698
    assert abs(err - reference) < ATOL


def test_basic_balanced():
    y_true = np.array([True, True, True, True, True, True, True, False])
    y_pred = np.array([0.491, -0.1, 0.64, 1.52, -0.23, -0.23, 1.579, 0.76])

    err = rmse(y_true, y_pred)
    reference = 0.8520359440774784
    assert abs(err - reference) < ATOL

    err = rmse(y_true, y_pred, balanced=True)
    reference = 0.81219216955952755
    assert abs(err - reference) < ATOL


def test_basic100():
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    y_pred = rng.randn(y_true.size)

    err = rmse(y_true, y_pred)
    reference = 1.4024162184449178
    assert abs(err - reference) < ATOL


def test_basic_vs_numpy():

    rng = np.random.RandomState(42)
    y_true = rng.randn(1000)
    y_pred = rng.randn(y_true.size)

    err = rmse(y_true, y_pred)
    reference = np.sqrt(((y_true - y_pred) ** 2.).mean())
    assert abs(err - reference) < ATOL
