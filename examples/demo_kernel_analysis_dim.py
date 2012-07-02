"""Example the effect of noise dimension on kernel analysis
Given a 1-D input that determines the decision boundary, we
add input dimensions that only contain noise.

The area under the kernel analysis curve increases as we add
noisy dims (the performance of the feautre space goes down).

Authors: Charles Cadieu <cadieu@mit.edu>
"""
import numpy as np
from matplotlib import pyplot as plt

from bangmetric import kanalysis

npoints = 300
Xsupport = np.arange(-1.,1.,.01)
X = np.random.uniform(-1,1,[npoints,1])
T = np.sign(X)
V = kanalysis(X,T)
auc = V.sum()
print 'Original kanalysis AUC = %2.2f'%auc

ndims = [1,9,19,39,79,199]

auc_ndims = []
V_ndims = []
for ndim in ndims:

    Xprime = np.hstack((X,np.random.uniform(-1,1,[npoints,ndim])))
    Vprime = kanalysis(Xprime,T)
    aucprime = Vprime.sum()
    print 'Added dims = %04d, kanalysis AUC = %2.2f'%(ndim,aucprime)
    auc_ndims.append(aucprime)
    V_ndims.append(Vprime)

plt.figure(1,figsize=(12,8))
plt.clf()
plt.plot(range(len(V)),V,'r-',label='X original, AUC=%2.2f'%auc)
for ind, ndim in enumerate(ndims):
    plt.plot(range(len(V_ndims[ind])),V_ndims[ind],
        label='X + %d dims, AUC=%2.2f'%(ndim,auc_ndims[ind]))
plt.axis([0,30,0,1.])
plt.legend()
plt.title('Kernel Analysis curves')
plt.xlabel('kPCA dimension')
plt.ylabel('loss (lower is better)')

plt.draw()
plt.show()
