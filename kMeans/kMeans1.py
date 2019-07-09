#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:45:50 2019

@author: 
"""
# Euclidean Distance Caculator
def dist ( a , b , ax = 1 ) :
    return np . linalg . norm ( a - b , axis = ax )

## matplotlib inline
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt . rcParams [ 'figure.figsize' ] = ( 16 , 9 )
plt . style . use ( 'ggplot' )

# Importing the dataset
data = pd . read_csv ( 'xclara.csv' )
print ( data . shape )
data . head ( )
# Getting the values and plotting it
f1 = data [ 'V1' ] . values
f2 = data [ 'V2' ] . values
X = np . array ( list ( zip ( f1 , f2 ) ) )
plt . scatter ( f1 , f2 , c = 'black' , s = 7 )

from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=5)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print(centroids) # From sci-kit learn

#Example 2

#We will generate a new dataset using make_blobs function.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.rcParams['figure.figsize'] = (16, 9)

# Creating a sample dataset with 4 clusters
X, y = make_blobs(n_samples=800, n_features=5, centers=5)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])


