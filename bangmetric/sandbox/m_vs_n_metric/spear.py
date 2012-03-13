#from scipy import stats
import numpy as np


def spearman(a1, a2):
    r1 = np.empty_like(a1)
    r1[a1.argsort()] = np.arange(a1.size)
    r2 = np.empty_like(a2)
    r2[a2.argsort()] = np.arange(a2.size)

    r1 -= r1.mean()
    r1_norm = np.linalg.norm(r1)
    assert r1_norm != 0
    r1 /= r1_norm

    r2 -= r2.mean()
    r2_norm = np.linalg.norm(r2)
    assert r2_norm != 0
    r2 /= r2_norm

    spear = np.dot(r1.ravel(), r2.ravel())
    #gt = stats.spearmanr(a1, a2)[0]
    #assert spear == gt, (spear, gt)

    return spear
