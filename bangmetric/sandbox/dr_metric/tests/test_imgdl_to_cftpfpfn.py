#!/usr/bin/env python

"""
Unit tests for function ``imgdl_to_cftpfpfn`` in module
``imgdl_to_cftpfpfn.py``.
"""

from nose.tools import raises
from ..imgdl_to_cftpfpfn import imgdl_to_cftpfpfn


@raises(AssertionError)
def test_non_equal_input_lists():
    gv_imgdl = [{'filename': '001.png', 'sha1': '12rd4356dg'}]
    gt_imgdl = []
    imgdl_to_cftpfpfn(gv_imgdl, gt_imgdl)
    return


@raises(AssertionError)
def test_empty_input_lists():
    gv_imgdl = []
    gt_imgdl = []
    imgdl_to_cftpfpfn(gv_imgdl, gt_imgdl)
    return


@raises(AssertionError)
def test_different_sha1s():
    gv_imgdl = [{'filename': '1.jpg', 'sha1': '12345qbcde'}]
    gt_imgdl = [{'filename': '2.jpg', 'sha1': '12345qbcdf'}]
    imgdl_to_cftpfpfn(gv_imgdl, gt_imgdl)
    return


@raises(AssertionError)
def test_all_sha1s_are_distinct():
    gv_imgdl = [{'filename': '1.jpg', 'sha1': '12345qbcde'},
                {'filename': '2.jpg', 'sha1': '12345qbpqr'}]
    gt_imgdl = [{'filename': '1.jpg', 'sha1': '12345qbcde'},
                {'filename': '2.jpg', 'sha1': '12345qbpqr'}]
    imgdl_to_cftpfpfn(gv_imgdl, gt_imgdl)
    return


@raises(AssertionError)
def test_missing_objects():
    gv_imgdl = [{'filename': '1.jpg', 'sha1': '12345qbcde', 'objects': []}]
    gt_imgdl = [{'filename': '2.jpg', 'sha1': '12345qbcde'}]
    imgdl_to_cftpfpfn(gv_imgdl, gt_imgdl)
    return


def test_empty_objects_fields():
    gv_imgdl = [{'filename': '1.jpg', 'sha1': '12345qbcde', 'objects': []}]
    gt_imgdl = [{'filename': '2.jpg', 'sha1': '12345qbcde', 'objects': []}]
    imgdl_to_cftpfpfn(gv_imgdl, gt_imgdl)
    return


@raises(AssertionError)
def test_zero_min_overlap():
    gv_imgdl = [{'filename': '1.jpg', 'sha1': '12345qbcde', 'objects': []}]
    gt_imgdl = [{'filename': '2.jpg', 'sha1': '12345qbcde', 'objects': []}]
    imgdl_to_cftpfpfn(gv_imgdl, gt_imgdl, min_overlap=0)
    return
