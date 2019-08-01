# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 08:40:13 2019

@author: Niket
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the mail dataset with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300)
ymeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[ymeans == 0, 0], X[ymeans == 0, 1], s=100, c='red', label = 'Cluster 1')
plt.scatter(X[ymeans == 1, 0], X[ymeans == 1, 1], s=100, c='blue', label = 'Cluster 2')
plt.scatter(X[ymeans == 2, 0], X[ymeans == 2, 1], s=100, c='green', label = 'Cluster 3')
plt.scatter(X[ymeans == 3, 0], X[ymeans == 3, 1], s=100, c='cyan', label = 'Cluster 4')
plt.scatter(X[ymeans == 4, 0], X[ymeans == 4, 1], s=100, c='magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income k$')
plt.ylabel('Speding score (1-100)')
plt.show()