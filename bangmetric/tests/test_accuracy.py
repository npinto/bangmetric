"""Test suite for the ``accuracy`` module"""

import numpy as np
from pytest import raises

from bangmetric import accuracy


def test_error_length():
    y_pred = np.array([False, True, True])
    y_true = np.array([True, False])
    raises(AssertionError, accuracy, y_true, y_pred)


def test_basic():
    y_true = np.array([True, False, True, True, False])
    y_pred = np.array([True, True, False, True, False])
    acc = accuracy(y_true, y_pred)
    reference = 3. / 5.
    assert acc == reference


def test_basic_balanced():
    y_true = np.array([True, True, True, False, False])
    y_pred = np.array([True, True, False, True, False])
    acc = accuracy(y_true, y_pred, balanced=True)
    reference = ((2. / 3.) + (1. / 2.)) / 2.
    assert acc == reference


def test_all_positives():
    y_true = np.ones((5), dtype=bool)
    y_pred = np.random.randn(y_true.size)
    y_pred -= y_pred.min()
    y_pred += 1.
    acc = accuracy(y_true, y_pred)
    assert acc == 1.0
    acc = accuracy(y_pred, y_true)
    assert acc == 1.0


def test_all_negatives():
    y_true = np.zeros((5), dtype=bool)
    y_pred = ~y_true
    acc = accuracy(y_true, y_pred)
    assert acc == 0.0


def test_error_non_finite():
    y_true = np.zeros((5), dtype=float)
    y_pred = np.ones_like(y_true)
    y_pred[0] = np.nan
    raises(AssertionError, accuracy, y_true, y_pred)
    y_pred[0] = np.inf
    raises(AssertionError, accuracy, y_true, y_pred)
