"""Test suite for the ``Fiji_metrics`` module"""

from os import environ
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from pytest import raises

from bangmetric.Fiji_metrics import isbi_metrics

EPSILON = 1e-4


def test_no_environment_variable():

    with raises(ValueError):
        # -- making sure the environment variable is
        #    either not there or wrong
        if "FIJI_EXE_PATH" in environ:
            environ["FIJI_EXE_PATH"] = "/no/path"

        # -- creating a fake prediction
        pred = np.eye(32).astype('f')
        pred = gaussian_filter(pred, 2.)
        pred -= pred.min()
        pred /= pred.max()
        pred = 1. - pred

        # -- creating a fake ground truth
        true = 1 - np.eye(32).astype(np.uint8)

        metrics = isbi_metrics(true, pred)


def test_simple_segmentation_with_provided_path():

    if "FIJI_EXE_PATH" not in environ:
        environ["FIJI_EXE_PATH"] = \
        "/home/poilvert/connectomics/Fiji-package/Fiji.app/fiji-linux64"

    # -- creating a fake prediction
    pred = np.eye(32).astype('f')
    pred = gaussian_filter(pred, 2.)
    pred -= pred.min()
    pred /= pred.max()
    pred = 1. - pred

    # -- creating a fake ground truth
    true = 1 - np.eye(32).astype(np.uint8)

    metrics = isbi_metrics(true, pred)

    assert np.abs(metrics['pixel_error'] - 0.010183) < EPSILON
    assert np.abs(metrics['rand_error'] - 0.020385) < EPSILON
    assert np.abs(metrics['warping_error'] - 0.0) < EPSILON
