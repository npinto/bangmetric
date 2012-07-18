# Authors: Nicolas Pinto <pinto@alum.mit.edu>
#
# License: BSD 3-clause


import numpy as np

from sklearn.preprocessing import LabelBinarizer

from bangmetric.rls_looe import rls_looe

EPSILON = 1e-6


def test_smoke_val_and_size():
    rng = np.random.RandomState(42)
    X = rng.randn(10, 4)
    Y = rng.randn(*X.shape)
    looe = rls_looe(X, Y, l2_regularizations=[0.1, 1, 10])
    assert abs(looe - 0.92153635747345852) < EPSILON
    looe, cv_values = rls_looe(X, Y,
                               l2_regularizations=[0.1, 1, 10],
                               return_cv_values=True)
    assert abs(looe - 0.92153635747345852) < EPSILON
    assert cv_values.shape == (10, 4, 3)


def test_simple_ova():
    rng = np.random.RandomState(42)
    n_features = 10
    n_classes = 4
    n_samples_per_class = 5
    n_samples = n_classes * n_samples_per_class

    X = rng.randn(n_samples, n_features)
    Y = np.array(range(n_classes) * n_samples_per_class)
    print LabelBinarizer
    lbin = LabelBinarizer(neg_label=-1, pos_label=+1)
    Y_ova = lbin.fit_transform(Y)
    for label in np.unique(Y):
        assert (Y==label).sum() == n_samples_per_class
        X[Y==label] += (label * 100)

    looe, cv_values = rls_looe(X, Y_ova,
                               return_cv_values=True)
    assert abs(looe - 0.5370459919085353) < EPSILON
    gt = np.array([[0.11319584, 0.05173503],
                   [0.11874747, 0.18956525]])
    assert abs((cv_values[1, 2:4, 4:6] - gt) < EPSILON).all()
