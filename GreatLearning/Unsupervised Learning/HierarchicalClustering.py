# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:21:36 2020

@author: Niket
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import zscore
import seaborn as sns

custData = pd.read_csv("Cust_Spend_Data.csv")
custData.head(10)

custDataAttr = custData.iloc[:, 2:]
custDataAttr.head()

custDataScaled = custDataAttr.apply(zscore)
custDataScaled.head()

sns.pairplot(custDataScaled, height= 2, aspect =2, diag_kind = 'kdc')

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'average')
model.fit(custDataScaled)

custDataAttr['labels'] = model.labels_
custDataAttr.head()

custDataClust = custDataAttr.groupby(['labels'])
custDataClust.mean()

from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist

z = linkage(custDataScaled, metric = 'euclidean', method = 'average')
c, coph_dists = cophenet(z, pdist(custDataScaled))

c

plt.figure(figsize = (10, 5))
plt.title('Agglomarative Hierrarchical Clustering Dendrogram')
plt.xlabel('sample Index')
plt.ylabel('Distancee')
dendrogram(z, leaf_rotation = 90, color_threshold= 40, leaf_font_size= 8.)
plt.tight_layout()

z = linkage(custDataScaled, metric = 'euclidean', method = 'complete')
c, coph_dists = cophenet(z, pdist(custDataScaled))


plt.figure(figsize = (10, 5))
plt.title('Agglomarative Hierrarchical Clustering Dendrogram')
plt.xlabel('sample Index')
plt.ylabel('Distancee')
dendrogram(z, leaf_rotation = 90, color_threshold= 40, leaf_font_size= 8.)
plt.tight_layout()


z = linkage(custDataScaled, metric = 'euclidean', method = 'ward')
c, coph_dists = cophenet(z, pdist(custDataScaled))


plt.figure(figsize = (10, 5))
plt.title('Agglomarative Hierrarchical Clustering Dendrogram')
plt.xlabel('sample Index')
plt.ylabel('Distancee')
dendrogram(z, leaf_rotation = 90, color_threshold= 40, leaf_font_size= 8.)
plt.tight_layout()

