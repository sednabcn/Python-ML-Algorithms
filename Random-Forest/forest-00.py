#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:03:24 2019

@author: sedna
 In comparison, the Random Forest algorithm randomly selects observations and features 
 to build several decision trees and then averages the results.
 Another difference is that „deep“ decision trees might suffer from overfitting.
 Random Forest prevents overfitting most of the time, by creating random subsets of the features and 
 building smaller trees using these subsets. Afterwards, it combines the subtrees. 
 Note that this doesn’t work every time and that it also makes the computation slower, 
 depending on how many trees your random forest builds.
 Important hyperparameters
 1. Increasing the Predictive Power
 a) n_estimators
 b) max_features
 c) min_sample_leaf of an internal node
2. Increase the models speed
n_jobs---number of jobs at the queue
random-state: replicable
oob_score (oob sampling cross-validation with the third part of the training dataset)
The main limitation of Random Forest is that a large number of trees can make the algorithm 
to slow and ineffective for real-time predictions. In general, these algorithms are fast to train, 
but quite slow to create predictions once they are trained. 
A more accurate prediction requires more trees, which results in 
a slower model. In most real-world applications the random forest algorithm is fast enough, 
but there can certainly be situations where run-time performance is important and other approaches would be preferred.
"""
import pandas as pd
import numpy as np

dataset=pd.read_csv('petrol_consumption.csv')
print(dataset.head())


# Preparing data for Training
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

"""
6. Evaluating the Algorithm

The last and final step of solving a machine learning problem is to evaluate the performance of the algorithm. 
For regression problems the metrics used to evaluate an algorithm are mean absolute error, mean squared error, 
and root mean squared error. Execute the following code to find these values:
"""
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

