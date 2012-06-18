import numpy as np
from bangmetric import kanalysis

# ----------------------------------------------------
# Example of execution on a sinc() function dataset
# ----------------------------------------------------
Xsupport = np.arange(-1., 1., .01)
X = np.random.uniform(-1, 1, [100, 1])
T = np.sign(np.cos(4 * np.pi * X))
Ttrue = np.sign(np.cos(4 * np.pi * Xsupport))
V = kanalysis(X, T)
auc = V.sum()
print 'kanalysis curve:', V
print 'kanalysis AUC:', auc

Xprime = np.cos(4 * np.pi * X)

Vprime = kanalysis(Xprime, T)
aucprime = Vprime.sum()
print 'kanalysis curve:', Vprime
print 'kanalysis AUC:', aucprime

from matplotlib import pyplot as plt
plt.figure(1, figsize=(16, 8))
plt.clf()
plt.subplot(1, 2, 1)
pos_inds = np.argwhere(T == 1.)[:, 0]
plt.scatter(X[pos_inds, :], T[pos_inds, :], marker='x', label='Positive Examples')
neg_inds = np.argwhere(T == -1.)[:, 0]
plt.scatter(X[neg_inds, :], T[neg_inds, :], marker='o', label='Negative Examples')
plt.plot(Xsupport, Ttrue, label='True Decision Function')
plt.scatter(X, X, marker='.', c='r', label='X (linear)')
plt.scatter(X, Xprime, marker='.', c='b', label='Xprime (cosine)')
plt.legend()
plt.title('Input function')

plt.subplot(1, 2, 2)
plt.plot(range(len(V)), V, 'r-', label='Kanalysis (X), AUC=%2.2f' % auc)
plt.plot(range(len(Vprime)), Vprime, 'b--', label='Kanalysis (Xprime), AUC=%2.2f' % aucprime)
plt.axis([0, 15, 0, 1.])
plt.legend()
plt.title('Kernel Analysis curves')
plt.xlabel('kPCA dimension')
plt.ylabel('loss (lower is better)')

plt.draw()
plt.show()
