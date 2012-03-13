"""Test suite for the ``correlation`` module"""

import numpy as np
from scipy import stats
from pytest import raises

from bangmetric import pearson, spearman

ATOL = 1e-6


def test_error_length():
    y_pred = np.array([False, True, True])
    y_true = np.array([True, False])
    raises(AssertionError, spearman, y_pred, y_true)
    raises(AssertionError, pearson, y_pred, y_true)


def test_basic():
    y_true = np.array([False, True, True, True, False, False, False, True])
    y_pred = np.array([0.491, -0.1, 0.64, 1.52, -0.23, -0.23, 1.579, 0.76])

    phi = pearson(y_true, y_pred)
    reference = 0.2225646085633681
    assert abs(phi - reference) < ATOL

    rho = spearman(y_true, y_pred)
    reference = 0.5714285714285714
    assert abs(rho - reference) < ATOL


def test_basic100():
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    y_pred = rng.randn(y_true.size)

    phi = pearson(y_true, y_pred)
    reference = -0.13642221217000247
    assert abs(phi - reference) < ATOL

    rho = spearman(y_true, y_pred)
    reference = -0.10796279627962799
    assert abs(rho - reference) < ATOL


def test_basic_vs_scipy():

    rng = np.random.RandomState(42)
    y_true = rng.randn(1000)
    y_pred = rng.randn(y_true.size)

    phi = pearson(y_true, y_pred)
    reference = stats.pearsonr(y_true, y_pred)[0]
    assert abs(phi - reference) < ATOL

    rho = spearman(y_true, y_pred)
    reference = stats.spearmanr(y_true, y_pred)[0]
    assert abs(rho - reference) < ATOL
