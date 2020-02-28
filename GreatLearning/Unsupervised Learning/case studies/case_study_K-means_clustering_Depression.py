# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:28:49 2020

@author: Niket
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

mydata = pd.read_csv("Depression.csv")

mydata.head()

mydata.drop('id', axis = 1, inplace = True)
mydata.info()

sns.pairplot(mydata, diag_kind = 'kde')

from scipy.stats import zscore
mydata_z = mydata.apply(zscore)

from scipy.spatial.distance import cdist
clusters = range(1, 10)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(mydata_z)
    prediction = model.predict(mydata)
    meanDistortions.append(sum(np.min(cdist(mydata, model.cluster_centers_, 'euclidean'), axis = 1))/mydata.shape[0])
    
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('selecting k with elbow method')

# k with 3
kmeans = KMeans(n_clusters=3, n_init=15, random_state = 1)
kmeans.fit(mydata_z)

centroids = kmeans.cluster_centers_
centroids

centroid_df = pd.DataFrame(centroids, columns = list(mydata))
centroid_df

df_labels = pd.DataFrame(kmeans.labels_, columns = list(['labels']))
df_labels['labels'] = df_labels['labels'].astype('category')

df_labeled = mydata.join(df_labels)

df_analysis = (df_labeled.groupby(['labels'], axis = 0)).head(4177)
df_analysis    

df_labeled['labels'].value_counts()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = Axes3D(fig, rect=[0,0,.95,1], elev = 20, azim=60)
kmeans.fit(mydata_z)
labels = kmeans.labels_
ax.scatter(mydata_z.iloc[:, 0], mydata_z.iloc[:, 1], mydata_z.iloc[:, 2], c = labels.astype(np.float), edgecolor =  'k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Length')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')
ax.set_title('3D plot of KMeans clustering')

# k = 2 
# final model
final_model = KMeans(n_clusters=2)
final_model.fit(mydata_z)
prediction = final_model.predict(mydata_z)

mydata["GROUP"] = prediction
print('Groups Assigned :\n')
mydata[["depression", "GROUP"]]

mydata.boxplot(by="GROUP", layout=(2, 4), figsize=(20,15))

mydata["simplicity"].corr(mydata["depression"])

plt.plot(mydata["simplicity"], mydata["depression"], 'bo')
z = np.polyfit(mydata["simplicity"],mydata["depression"],1)
p = np.poly1d(z)
plt.plot(mydata["simplicity"], p(mydata["simplicity"]), "r--")
