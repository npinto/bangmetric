#!/usr/bin/env python

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>

# Licence: BSD

# TODO: refactor

import numpy as np

import skimage.io as io
io.use_plugin('freeimage')

from os import path, environ
import inspect
from tempfile import mkdtemp
from shutil import rmtree
from skimage.exposure import rescale_intensity

import re

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -- absolute paths to the Fiji scripts
base_path = path.split(path.abspath(__file__))[0]
PX_SCRIPT_PATH = path.abspath(path.join(base_path, "px_err_cli.bsh"))
RD_SCRIPT_PATH = path.abspath(path.join(base_path, "rd_err_cli.bsh"))
WP_SCRIPT_PATH = path.abspath(path.join(base_path, "wp_err_cli.bsh"))
WARP_CODE_PATH = path.abspath(path.join(base_path, "warp_cli.bsh"))

# -- default paths to find the Fiji executable
ENV_VAR = "FIJI_EXE_PATH"
DEFAULT_FIJI_EXE_PATH = "/path/to/fiji-linux64"


def pixel_error(y_true, y_pred, th_min=0., th_max=0.9, th_inc=0.1):
    """XXX: docstring"""

    out = _metrics(y_true, y_pred,
                   pixel_error=True,
                   rand_error=False,
                   warping_error=False,
                   th_min=th_min,
                   th_max=th_max,
                   th_inc=th_inc)
    return out['pixel_error']


def rand_error(y_true, y_pred, th_min=0., th_max=0.9, th_inc=0.1):
    """XXX: docstring"""

    out = _metrics(y_true, y_pred,
                   pixel_error=False,
                   rand_error=True,
                   warping_error=False,
                   th_min=th_min,
                   th_max=th_max,
                   th_inc=th_inc)
    return out['rand_error']


def warp_error(y_true, y_pred, th_min=0., th_max=0.9, th_inc=0.1):
    """XXX: docstring"""

    out = _metrics(y_true, y_pred,
                   pixel_error=False,
                   rand_error=False,
                   warping_error=True,
                   th_min=th_min,
                   th_max=th_max,
                   th_inc=th_inc)
    return out['warping_error']


def warp_2d(y_true, y_pred, y_out_fname='y_out.tif', thr=0., radius=20):

    program, true_tmp_file, pred_tmp_file, tmpdir = \
            _prepare_arrays(y_true, y_pred)

    script = WARP_CODE_PATH

    y_out_fname_full_path = path.join(tmpdir, y_out_fname)

    cmdline = "%s %s %s %s %s %s %s %s" % (program, '--headless', script,
            true_tmp_file, pred_tmp_file,
            y_out_fname_full_path, thr, radius)

    return_code, stdout, stderr = _call_capture_output(cmdline)

    if return_code == 0:
        y_out = io.imread(y_out_fname_full_path, as_grey=True, plugin='freeimage')
        y_out = (y_out > 0)
    else:
        print 'An error occured while executing command :'
        print '%s' % cmdline
        raise ExecError("return code %s" % return_code)

    rmtree(tmpdir)

    return y_out


def _metrics(y_true, y_pred,
             pixel_error=True,
             rand_error=True,
             warping_error=True,
             th_min=0.,
             th_max=0.9,
             th_inc=0.1):
    """
    Computes the metrics from the ISBI challenge, using Fiji.

    Parameters
    ----------

    `y_true`: 2D array
        ground truth annotations

    `y_pred`: 2D array
        predictions

    `pixel_error`: bool
        whether or not to compute the Pixel Error as defined in ISBI

    `rand_error`: bool
        whether or not to compute the Rand Error as defined in ISBI

    `warping_error`: bool
        whether or not to compute the Warping Error as defined in ISBI

    `th_min`: float
        minimum threshold for the threshold sweep

    `th_max`: float
        maximum threshold for the threshold sweep

    `th_inc`: float
        threshold increment for the threshold sweep

    Returns
    -------

    `metrics`: a dictionnary containing the values for the metrics
        as computed by Fiji. Dictionnary keys are 'pixel_error',
        'rand_error' and 'warping_error'

    """

    assert 0. <= th_min and th_min <= 1.
    assert 0. <= th_max and th_max <= 1.
    assert 0. <= th_inc and th_inc <= 1.
    assert th_min <= th_max

    # -- special cases
    if not pixel_error and \
       not rand_error and \
       not warping_error:
        return {}

    program, true_tmp_file, pred_tmp_file, tmpdir = \
            _prepare_arrays(y_true, y_pred)

    metrics = {}

    if pixel_error:

        script = PX_SCRIPT_PATH

        cmdline = "%s %s %s %s %s %s %s %s" % (program, '--headless', script,
                                               true_tmp_file, pred_tmp_file,
                                               th_min, th_max, th_inc)

        return_code, stdout, stderr = _call_capture_output(cmdline)

        if return_code == 0:
            px_metric = _parse_stdout(stdout, metric='pixel')
            #print px_metric
            metrics['pixel_error'] = float(px_metric)
        else:
            print 'An error occured while executing command :'
            print '%s' % cmdline
            raise ExecError("return code %s" % return_code)

    if rand_error:

        script = RD_SCRIPT_PATH

        cmdline = "%s %s %s %s %s %s %s %s" % (program, '--headless', script,
                                               true_tmp_file, pred_tmp_file,
                                               th_min, th_max, th_inc)

        return_code, stdout, stderr = _call_capture_output(cmdline)

        if return_code == 0:
            rd_metric = _parse_stdout(stdout, metric='rand')
            #print rd_metric
            metrics['rand_error'] = float(rd_metric)
        else:
            print 'An error occured while executing command :'
            print '%s' % cmdline
            raise ExecError("return code %s" % return_code)

    if warping_error:

        script = WP_SCRIPT_PATH

        cmdline = "%s %s %s %s %s %s %s %s" % (program, '--headless', script,
                                               true_tmp_file, pred_tmp_file,
                                               th_min, th_max, th_inc)

        return_code, stdout, stderr = _call_capture_output(cmdline)

        if return_code == 0:
            wp_metric = _parse_stdout(stdout, metric='warping')
            #print wp_metric
            metrics['warping_error'] = float(wp_metric)
        else:
            print 'An error occured while executing command :'
            print '%s' % cmdline
            raise ExecError("return code %s" % return_code)

    rmtree(tmpdir)

    # -- returning a dictionnary containing the metrics
    return metrics


def _parse_stdout(output, metric='pixel'):

    assert metric in ['pixel', 'rand', 'warping']

    if metric == 'pixel':
        regex = r"  Minimum pixel error: (\d+.\d+)"
    elif metric == 'rand':
        regex = r"  Minimum Rand error: (\d+.\d+)"
    elif metric == 'warping':
        regex = r"  Minimum warping error: (\d+.\d+)"

    lines = output.split("\n")

    for line in lines:
        match = re.match(regex, line)
        if match:
            # extract metric value
            metric_value = float(match.groups()[0])

    return metric_value


def _prepare_arrays(y_true, y_pred):

    # -- figuring out where the Fiji executable is
    #log.warn("looking for %s environment variable..." % ENV_VAR)
    if ENV_VAR not in environ:

        if not path.exists(DEFAULT_FIJI_EXE_PATH):
            raise ValueError(
                "Could not find the path to the Fiji executable!"
                "Please set the %s environment variable or set"
                " the default path in %s" % (ENV_VAR,
                path.abspath(inspect.getfile(inspect.currentframe()))))
        else:
            FIJI_EXE_PATH = DEFAULT_FIJI_EXE_PATH

    else:
        FIJI_EXE_PATH = environ.get(ENV_VAR)

    if not path.exists(FIJI_EXE_PATH):
        raise ValueError("%s does not exist" % FIJI_EXE_PATH)

    assert y_true.ndim == 2
    assert y_true.ndim == y_pred.ndim

    # -- making sure that the arrays are in the proper ranges and have the
    # proper dtype
    y_true = 255 * (y_true > 0).astype(np.uint8)
    y_pred = rescale_intensity(y_pred.astype(np.float32))

    # -- saving the arrays as tif images in a temp dir
    tmpdir = mkdtemp()
    true_tmp_file = path.join(tmpdir, 'true.tif')
    pred_tmp_file = path.join(tmpdir, 'pred.tif')
    io.imsave(true_tmp_file, y_true, plugin='freeimage')
    io.imsave(pred_tmp_file, y_pred, plugin='freeimage')

    # -- now we call the Fiji program to compute the metrics
    program = FIJI_EXE_PATH

    return (program, true_tmp_file, pred_tmp_file, tmpdir)


def _call_capture_output(cmdline, cwd=None, error_on_nonzero=True):
    """
    Returns
    -------
    a tuple (return code, stdout_data, stderr_data)

    Reference
    ---------
    from Andreas Klochner
    https://github.com/inducer/pytools/blob/master/pytools/prefork.py#L39
    modified by Nicolas Poilvert, 2011
    """
    from subprocess import Popen, PIPE
    import shlex
    try:
        args = shlex.split(cmdline)
        popen = Popen(args, cwd=cwd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout_data, stderr_data = popen.communicate()
        if error_on_nonzero and popen.returncode:
            raise ExecError("status %d invoking '%s': %s"
                    % (popen.returncode, "".join(cmdline), stderr_data))
        return popen.returncode, stdout_data, stderr_data
    except OSError, e:
        raise ExecError("error invoking '%s': %s"
                % ("".join(cmdline), e))


class ExecError(OSError):
    pass
