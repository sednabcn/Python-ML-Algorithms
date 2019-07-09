#!/usr/bin/env python

#F-TEST
sklearn.feature_selection.f_regression
sklearn.feature_selection.f_classif

# MUTUAL INFORMATION
sklearn.feature_selection.mututal_info_regression 
sklearn.feature_selection.mututal_info_classif

# VARIANCE TRESHOLD
"""
Variance Threshold doesn’t consider the relationship of features with the target variable.
"""
sklearn.feature_selection.VarianceThreshold

# WRAPPER METHODS
"""
Wrapper Methods generate models with a subsets of feature and gauge their model performances.
"""

# FORWARD SEARCH

"""
This method allows you to search for the best feature w.r.t model performance and add them to your feature subset one after the other.
For data with n features,

->On first round ‘n’ models are created with individual feature and the best predictive feature is selected.

->On second round, ‘n-1’ models are created with each feature and the previously selected feature.

->This is repeated till a best subset of ‘m’ features are selected.

"""

#RECURSIVE FEATURE ELIMINATION

"""
For data with n features,

->On first round ‘n-1’ models are created with combination of all features except one. The least performing feature is removed

-> On second round ‘n-2’ models are created by removing another feature.

Wrapper Methods promises you a best set of features with a extensive greedy search.

But the main drawbacks of wrapper methods is the sheer amount of models that needs to be trained. It is computationally very expensive and is infeasible with large number of features.
"""

#EMBEDDED METHODS

"""
Feature selection can also be acheived by the insights provided by some Machine Learning models.
"""

Lasso Linear Regression
Tree based models
