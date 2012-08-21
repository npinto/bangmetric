"""Test suite for the ``Fiji_metrics`` module"""

from os import environ, path
from scipy import misc
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pytest

from bangmetric.wildwest.isbi12 import pixel_error, rand_error, warp_error
from bangmetric.wildwest.isbi12 import warp2d

MYPATH = path.abspath(path.dirname(__file__))

EPSILON = 1e-4


def test_no_environment_variable():

    backup = environ["FIJI_EXE_PATH"]
    environ["FIJI_EXE_PATH"] = "/no/path"

    with pytest.raises(ValueError):
        # -- creating a fake prediction
        y_pred = np.eye(32).astype('f')
        y_pred = gaussian_filter(y_pred, 2.)
        y_pred -= y_pred.min()
        y_pred /= y_pred.max()
        y_pred = 1. - y_pred

        # -- creating a fake ground truth
        y_true = 1 - np.eye(32).astype(np.uint8)

        pixel_error(y_true, y_pred)

    environ["FIJI_EXE_PATH"] = backup


@pytest.mark.skipif("environ.get('FIJI_EXE_PATH') is None")
def test_simple_segmentation_with_provided_path():

    # -- creating a fake y_prediction
    y_pred = np.eye(32).astype('f')
    y_pred = gaussian_filter(y_pred, 2.)
    y_pred -= y_pred.min()
    y_pred /= y_pred.max()
    y_pred = 1. - y_pred

    # -- creating a fake ground truth
    y_true = 1 - np.eye(32).astype(np.uint8)

    pe = pixel_error(y_true, y_pred)
    re = rand_error(y_true, y_pred)
    we = warp_error(y_true, y_pred)

    assert np.abs(pe - 0.010183) < EPSILON
    assert np.abs(re - 0.020385) < EPSILON
    assert np.abs(we - 0.0) < EPSILON


@pytest.mark.skipif("environ.get('FIJI_EXE_PATH') is None")
def test_simple_unique_rand_value():

    # -- creating a fake y_prediction
    y_pred = np.eye(32).astype('f')
    y_pred = gaussian_filter(y_pred, 2.)
    y_pred -= y_pred.min()
    y_pred /= y_pred.max()
    y_pred = 1. - y_pred

    # -- creating a fake ground truth
    y_true = 1 - np.eye(32).astype(np.uint8)

    re = rand_error(y_true, y_pred, th_min=0.6, th_max=0.7)

    assert np.abs(re - 0.13515) < EPSILON


def test_simple_warp2d():

    # -- read in y_pred and y_true from PNG images
    y_pred = misc.imread(path.join(MYPATH, 'y_pred128.png'), flatten=True)
    y_true = misc.imread(path.join(MYPATH, 'y_true128.png'), flatten=True)
    gt = misc.imread(path.join(MYPATH, 'y_warp128.png'), flatten=True) > 0

    # -- compute the "warped annotations"
    gv = warp2d(y_true, y_pred)
    print gv

    assert (abs(gt - gv) < EPSILON).all()
