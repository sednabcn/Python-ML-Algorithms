#!/usr/bin/env python

import numpy as np
# Dataset
n_samples, n_features = 100, 2
mean0, mean1 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])
np.random.seed(42)
X0 = np.random.multivariate_normal(mean0, Cov, n_samples)
X1 = np.random.multivariate_normal(mean1, Cov, n_samples)
X = np.vstack([X0, X1])
y = np.array([0] * X0.shape[0] + [1] * X1.shape[0])

"""
Ridge linear Support Vector Machine (L2-regularization)
Support Vector Machine seek for separating hyperplane with maximum margin to enforce robustness against noise.
Like logistic regression it is a discriminative method that only focuses of predictions.
Here we present the non separable case of Maximum Margin Classifiers with ±1 coding (ie.: y i {−1, +1})
So linear SVM is closed to Ridge logistic regression, using the hinge loss instead of the logistic loss. Both will provide
very similar predictions.
"""





from sklearn import svm
svmlin = svm.LinearSVC()
# Remark: by default LinearSVC uses squared_hinge as loss
svmlin.fit(X, y)
y_pred_svmlin = svmlin.predict(X)
errors = y_pred_svmlin != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_svmlin)))
print(svmlin.coef_)

"""Lasso linear Support Vector Machine (L1-regularization)
Linear SVM for classification (also called SVM-C or SVC) with l1-regularization
∑︀ N
min F Lasso linear SVM (w) = λ ||w|| 1 + Ci ξ i 

with ∀i y i (w · x i ) ≥ 1 − ξ i
"""
svmlinl1 = svm.LinearSVC(penalty='l1', dual=False)
# Remark: by default LinearSVC uses squared_hinge as loss
svmlinl1.fit(X, y)
y_pred_svmlinl1 = svmlinl1.predict(X)
errors = y_pred_svmlinl1 != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_svmlinl1)))
print(svmlinl1.coef_)
