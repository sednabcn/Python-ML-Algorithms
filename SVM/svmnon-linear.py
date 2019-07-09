#!/usr/bin/env python
"""
#Support Vector Machines (SVM)

SVM are based kernel methods require only a user-specified kernel function K(x i , x j ), i.e., a similarity function over
pairs of data points (x i , x j ) into kernel (dual) space on which learning algorithms operate linearly, i.e. every operation
on points is a linear combination of K(x i , x j ).
Outline of the SVM algorithm:
1. Map points x into kernel space using a kernel function: x → K(x, .).
2. Learning algorithms operate linearly by dot product into high-kernel space K(., x i ) · K(., x j ).
• Using the kernel trick (Mercer’s Theorem) replace dot product in hgh dimensional space by a simpler
operation such that K(., x i ) · K(., x j ) = K(x i , x j ). Thus we only need to compute a similarity measure
for each pairs of point and store in a N × N Gram matrix.
• Finally, The learning process consist of estimating the α i of the decision function that maximises the hinge
loss (of f (x)) plus some penalty when applied on all training points.

             N
f (x) = sign(︃∑α i y i K(x i , x))︃ ︁
             i
3. Predict a new point x using the decision function.

#Gaussian kernel (RBF, Radial Basis Function):

One of the most commonly used kernel is the Radial Basis Function (RBF) Kernel. For a pair of points x i , x j the RBF
kernel is defined as:

K(x i , x j ) = exp −(︂gamma‖x i − x j ‖)^2
"""
#Non linear SVM also exists for regression problems.
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
# dataset
X, y = datasets.make_classification(n_samples=10, n_features=2,n_redundant=0,
n_classes=2,
random_state=1,
shuffle=False)
clf = SVC(kernel='rbf', gamma='auto')
clf.fit(X, y)
print("#Errors: %i" % np.sum(y != clf.predict(X)))
clf.decision_function(X)
# Usefull internals:
# Array of support vectors
clf.support_vectors_
# indices of support vectors within original X
np.all(X[clf.support_,:] == clf.support_vectors_)
