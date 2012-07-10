# Authors: Nicolas Pinto <pinto@alum.mit.edu>
#          Charles Cadieu <cadieu@mit.edu>
#
# License: BSD 3-clause


import numpy as np

from bangmetric.kernel_analysis import kanalysis, kanalysis_K

EPSILON = 1e-6
QUANTILES = [0.1, 0.5, 0.9]


def test_smoke_size():
    rng = np.random.RandomState(42)
    X = rng.randn(10, 4)
    Y = rng.randn(*X.shape)
    gv = kanalysis(X, Y)
    assert gv.size == len(X) + 1


def test_simple1d():
    rng = np.random.RandomState(42)
    X = rng.randn(8, 4)
    Y = rng.randn(8)
    gv = kanalysis(X, Y)
    assert abs(gv[0] - 1) < EPSILON
    gt = np.array([
        1.00000000e+00, 9.71468057e-01, 5.59939220e-01,
        5.20056617e-01, 5.04160095e-01, 3.38182408e-01,
        3.12550170e-01, 1.50530684e-30, 1.03692068e-30,
    ])
    assert (abs(gv - gt) < EPSILON).all()


def test_simple2d():
    rng = np.random.RandomState(42)
    X = rng.randn(8, 4)
    Y = rng.randn(*X.shape)
    gv = kanalysis(X, Y)
    assert abs(gv[0] - 1) < EPSILON
    gt = np.array([
        1.00000000e+00, 8.36576988e-01, 7.62676568e-01,
        6.14490305e-01, 3.61717470e-01, 2.32448139e-01,
        1.12094204e-01, 9.04488924e-31, 7.08820460e-31,
    ])
    assert (abs(gv - gt) < EPSILON).all()


def test_sinc():
    rng = np.random.RandomState(42)
    X = rng.uniform(-1, 1, [100, 1])
    Y_true = np.sign(np.cos(4. * np.pi * X))

    ka = kanalysis(X, Y_true, quantiles=QUANTILES)
    assert abs(ka[0] - 1) < EPSILON
    auc = ka.mean()
    assert abs(ka[1] - 0.97858424670806643) < EPSILON
    assert abs(ka[6] - 0.77621800324925161) < EPSILON
    assert abs(auc - 0.26515079212133352) < EPSILON

    X2 = np.cos(4*np.pi*X)
    ka2 = kanalysis(X2, Y_true, quantiles=QUANTILES)
    assert abs(ka2[0] - 1) < EPSILON
    auc2 = ka2.mean()
    assert abs(ka2[2] - 0.18664740171388722) < EPSILON
    assert abs(ka2[8] - 0.08675467229138914) < EPSILON
    assert abs(auc2 - 0.054521533910044662) < EPSILON


def test_sinc_n_components():
    rng = np.random.RandomState(42)
    X = rng.uniform(-1, 1, [100, 1])
    Y_true = np.sign(np.cos(4. * np.pi * X))

    ka_gt = kanalysis(X, Y_true, n_components=len(X), quantiles=QUANTILES)
    assert abs(ka_gt[0] - 1) < EPSILON
    auc_gt = ka_gt.mean()
    assert abs(ka_gt[1] - 0.97858424670806643) < EPSILON
    assert abs(ka_gt[6] - 0.77621800324925161) < EPSILON
    assert abs(auc_gt - 0.26515079212133352) < EPSILON

    n_components = 10
    ka_gv = kanalysis(X, Y_true, n_components=n_components, quantiles=QUANTILES)
    assert abs(ka_gv[0] - 1) < EPSILON
    assert (abs(ka_gv - ka_gt[:n_components + 1]) < EPSILON).all()


def test_sinc_linear_kernel():
    rng = np.random.RandomState(42)
    X = rng.uniform(-1, 1, [100, 1])
    Y_true = np.sign(np.cos(4. * np.pi * X))

    K = np.dot(X, X.T)
    ka_gt = kanalysis_K(K, Y_true, n_components=len(X))
    assert abs(ka_gt[0] - 1) < EPSILON
    auc_gt = ka_gt.mean()
    assert abs(ka_gt[1] - 0.97911955529177741) < EPSILON
    assert abs(ka_gt[6] - 0.97911955529177741) < EPSILON
    assert abs(auc_gt - 0.97932629236809443) < EPSILON


def test_not_full_rank():
    rng = np.random.RandomState(42)
    rank = 4
    X = rng.randn(100, rank)
    K = np.dot(X, X.T)
    Y = 2. * rng.randn(len(X)) - 1
    gv = kanalysis_K(K, Y)
    assert abs(gv[-1] - 0) > EPSILON
    assert gv[rank] != gv[rank - 1]
    assert gv[rank] == gv[rank + 1]
