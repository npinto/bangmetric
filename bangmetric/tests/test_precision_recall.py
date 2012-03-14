"""Test suite for the ``precision_recall`` module"""

import warnings
import numpy as np
from pytest import raises

from bangmetric import average_precision
from bangmetric import precision
from bangmetric import recall


def test_incompatible_input_dimensions_ap():
    y_pred = np.array([0.1, 0.2, 0.3])
    y_true = np.array([1., -1.])
    raises(AssertionError,
    average_precision, y_true, y_pred)


def test_incompatible_input_dimensions_precision():
    y_pred = np.array([0.1, 0.2, 0.3])
    y_true = np.array([1., -1.])
    raises(AssertionError,
    precision, y_true, y_pred)


def test_incompatible_input_dimensions_recall():
    y_pred = np.array([0.1, 0.2, 0.3])
    y_true = np.array([1., -1.])
    raises(AssertionError,
    recall, y_true, y_pred)


def test_wrong_integration_method():
    y_pred = np.array([0.87, 0.21, 0.35])
    y_true = np.array([1., -1., 1.])
    raises(AssertionError,
    average_precision, y_true, y_pred, integration='dummy')


def test_equal_predictions_error_precision():
    y_pred = np.array([0.35, 0.35, 0.35])
    y_true = np.array([1., -1., 1.])
    raises(ValueError,
    precision, y_true, y_pred)


def test_equal_predictions_error_recall():
    y_pred = np.array([0.35, 0.35, 0.35])
    y_true = np.array([1., -1., 1.])
    raises(ValueError,
    recall, y_true, y_pred)


def test_equal_predictions_error_trapz():
    y_pred = np.array([0.35, 0.35, 0.35])
    y_true = np.array([1., -1., 1.])
    raises(ValueError,
    average_precision, y_true, y_pred, integration='trapz')


def test_equal_predictions_error_voc2010():
    y_pred = np.array([0.42, 0.42, 0.42])
    y_true = np.array([1., -1., 1.])
    raises(ValueError,
    average_precision, y_true, y_pred, integration='voc2010')


def test_equal_predictions_error_voc2007():
    y_pred = np.array([0.07, 0.07, 0.07])
    y_true = np.array([1., -1., 1.])
    raises(ValueError,
    average_precision, y_true, y_pred, integration='voc2007')


def test_user_warning_precision():
    y_pred = np.array([0.04, 0.04, 0.10])
    y_true = np.array([1., -1., 1.])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        precision(y_true, y_pred)
        assert issubclass(w[-1].category, UserWarning)
        assert "sorting method used" in str(w[-1].message)


def test_user_warning_recall():
    y_pred = np.array([0.04, 0.04, 0.10])
    y_true = np.array([1., -1., 1.])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        recall(y_true, y_pred)
        assert issubclass(w[-1].category, UserWarning)
        assert "sorting method used" in str(w[-1].message)


def test_user_warning_trapz():
    y_pred = np.array([0.04, 0.04, 0.10])
    y_true = np.array([1., -1., 1.])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        average_precision(y_true, y_pred, integration='trapz')
        assert issubclass(w[-1].category, UserWarning)
        assert "sorting method used" in str(w[-1].message)


def test_user_warning_voc2010():
    y_pred = np.array([0.04, 0.04, 0.10])
    y_true = np.array([1., -1., 1.])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        average_precision(y_true, y_pred, integration='voc2010')
        assert issubclass(w[-1].category, UserWarning)
        assert "sorting method used" in str(w[-1].message)


def test_user_warning_voc2007():
    y_pred = np.array([0.04, 0.04, 0.10])
    y_true = np.array([1., -1., 1.])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        average_precision(y_true, y_pred, integration='voc2007')
        assert issubclass(w[-1].category, UserWarning)
        assert "sorting method used" in str(w[-1].message)


def test_perfect_pos_predictions_trapz():
    y_pred = np.array([0.92, 0.99, 0.97])
    y_true = np.array([1., 1., 1.])
    ap = average_precision(y_true, y_pred, integration='trapz')
    reference = 1.
    assert abs(ap - reference) < 1e-6


def test_perfect_pos_predictions_voc2010():
    y_pred = np.array([0.92, 0.99, 0.97])
    y_true = np.array([1., 1., 1.])
    ap = average_precision(y_true, y_pred, integration='voc2010')
    reference = 1.
    assert abs(ap - reference) < 1e-6


def test_non_trivial_precision_trapz():
    y_pred = np.array([0.25, 0.45, 0.60, 0.90])
    y_true = np.array([1., 1., 0., 1.])
    prec = precision(y_true, y_pred)
    reference = np.array([1., 0.5, 2. / 3., 0.75])
    assert (np.abs(prec - reference)).sum() < 1e-6


def test_non_trivial_recall_trapz():
    y_pred = np.array([0.25, 0.45, 0.60, 0.90])
    y_true = np.array([1., 1., 0., 1.])
    rec = recall(y_true, y_pred)
    reference = np.array([1. / 3., 1. / 3., 2. / 3., 1.])
    assert (np.abs(rec - reference)).sum() < 1e-6


def test_non_trivial_ap_trapz():
    y_pred = np.array([0.25, 0.45, 0.60, 0.90])
    y_true = np.array([1., 1., 0., 1.])
    ap = average_precision(y_true, y_pred, integration='trapz')
    reference = 0.7638888888888888
    assert abs(ap - reference) < 1e-6


def test_non_trivial_ap_voc2010():
    y_pred = np.array([0.25, 0.45, 0.60, 0.90])
    y_true = np.array([1., 1., 0., 1.])
    ap = average_precision(y_true, y_pred, integration='voc2010')
    reference = 0.8333333333333333
    assert abs(ap - reference) < 1e-6


def test_non_trivial_perfect_ap_trapz():
    y_pred = np.array([0.82, 0.75, 0.60, 0.90])
    y_true = np.array([1., 1., 0., 1.])
    ap = average_precision(y_true, y_pred, integration='trapz')
    reference = 1.
    assert abs(ap - reference) < 1e-6


def test_non_trivial_perfect_ap_voc2010():
    y_pred = np.array([0.82, 0.75, 0.60, 0.90])
    y_true = np.array([1., 1., 0., 1.])
    ap = average_precision(y_true, y_pred, integration='voc2010')
    reference = 1.
    assert abs(ap - reference) < 1e-6
