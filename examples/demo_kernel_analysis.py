"""Example of kernel analysis on a cosine() function dataset

Authors: Charles Cadieu <cadieu@mit.edu>
"""
import numpy as np
from matplotlib import pyplot as plt

from bangmetric import kanalysis

# Create some data
# randomly sample a 1-D space between -1 and 1
X = np.random.uniform(-1,1,[100,1])
# our class boundary is determined by the sign of the cosine function:
T = np.sign(np.cos(4 * np.pi * X))

# Compute the kernel analysis on this space/boundary:
V = kanalysis(X, T)
auc = V.sum()
print 'Original space: kanalysis AUC =', auc

# Create a transformation of the original X space
# Note that for this demo, the function is matched to the decision boundary
Xprime = np.cos(4 * np.pi * X)

Vprime = kanalysis(Xprime, T)
aucprime = Vprime.sum()
print 'Transformed space: kanalysis AUC =', aucprime

plt.figure(1,figsize=(8,12))
plt.clf()
plt.subplot(3,1,1)
pos_inds = np.argwhere(T==1.)[:,0]
plt.scatter(X[pos_inds,:],np.zeros_like(X[pos_inds,:]),marker='x',label='Positive Examples',s=100)
neg_inds = np.argwhere(T==-1.)[:,0]
plt.scatter(X[neg_inds,:],np.zeros_like(X[neg_inds,:]),marker='o',label='Negatives Examples',s=100)
plt.legend()
plt.xlabel('X space')
plt.ylabel('N/A')
plt.title('X Input function')
plt.subplot(3,1,2)
pos_inds = np.argwhere(T==1.)[:,0]
plt.scatter(Xprime[pos_inds,:],np.zeros_like(Xprime[pos_inds,:]),marker='x',label='Positive Examples',s=100)
neg_inds = np.argwhere(T==-1.)[:,0]
plt.scatter(Xprime[neg_inds,:],np.zeros_like(Xprime[neg_inds,:]),marker='o',label='Negatives Examples',s=100)
plt.legend()
plt.xlabel('X space')
plt.ylabel('N/A')
plt.title('Xprime Input function')

#plt.figure(2,figsize=(16,8))
#plt.clf()
plt.subplot(3,1,3)
plt.plot(range(len(V)),V,'r-',label='Kanalysis (X), AUC=%2.2f'%auc)
plt.plot(range(len(Vprime)),Vprime,'b--',label='Kanalysis (Xprime), AUC=%2.2f'%aucprime)
plt.axis([0,15,0,1.])
plt.legend()
plt.title('Kernel Analysis curves')
plt.xlabel('kPCA dimension')
plt.ylabel('loss (lower is better)')

plt.draw()
plt.show()
