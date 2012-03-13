"""Test suite for the ``accuracy`` module"""

import numpy as np
from pytest import raises

from bangmetric import accuracy


def test_error_length():
    y_pred = np.array([False, True, True])
    y_true = np.array([True, False])
    raises(AssertionError, accuracy, y_pred, y_true)


def test_basic():
    y_pred = np.array([True, False, True, True, False])
    y_true = np.array([True, True, False, True, False])
    acc = accuracy(y_true, y_pred)
    reference = 3. / 5.
    assert acc == reference


def test_all_positives():
    y_true = np.ones((5), dtype=bool)
    y_pred = np.random.randn(y_true.size)
    y_pred -= y_pred.min()
    y_pred += 1.0
    acc = accuracy(y_true, y_pred)
    assert acc == 1.0
    acc = accuracy(y_pred, y_true)
    assert acc == 1.0


def test_all_negatives():
    y_true = np.zeros((5), dtype=bool)
    y_pred = ~y_true
    acc = accuracy(y_pred, y_true)
    assert acc == 0.0
