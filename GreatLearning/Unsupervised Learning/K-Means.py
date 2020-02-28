# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:17:55 2020

@author: Niket
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from scipy.stats import zscore

# import sklearn.metrics

tech_supp_df = pd.read_csv('technical_support_data-2.csv')
tech_supp_df.dtypes

tech_supp_df.shape

tech_supp_df.head()

techSuppAttr = tech_supp_df.iloc[:, 1:]
techSuppScaled = techSuppAttr.apply(zscore)
sns.pairplot(techSuppScaled, diag_kind = "kde")

from scipy.spatial.distance import cdist
clusters = range(1,10)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters = k)
    model.fit(techSuppScaled)
    predictions = model.predict(techSuppScaled)
    meanDistortions.append(sum(np.min(cdist(techSuppScaled, model.cluster_centers_, 'euclidean'), axis = 1)) / techSuppScaled.shape[0])

plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Distortion')
plt.title('Selecting k with the Elbow Method')

# let us first start with k = 3
final_model = KMeans(3)
final_model.fit(techSuppScaled)
prediction = final_model.predict(techSuppScaled)

# Append the prediction
tech_supp_df["Group"] = prediction
techSuppScaled["Group"] = prediction
print("Group Assigned : \n")
tech_supp_df.head()


# analyze the distribution of the data among the two groups (K = 3). One of the most informative visual tool is boxplot.
techSuppClust = tech_supp_df.groupby(["Group"])
techSuppClust.mean()

techSuppScaled.boxplot(by = "Group", layout = (2, 4), figsize = (15, 10))

# let us next try with k = 5, the next elbow point

# let us first start with k =5
final_model = KMeans(5)
final_model.fit(techSuppScaled)
prediction = final_model.predict(techSuppScaled)

# Append the prediction
tech_supp_df["Group"] = prediction
techSuppScaled["Group"] = prediction
print("Groups Assigned : \n")
tech_supp_df.head()

techSuppClust = tech_supp_df.groupby(["Group"])
techSuppClust.mean()

techSuppClust.boxplot(by = "Group", layout = (2, 4), figsize = (20, 10))