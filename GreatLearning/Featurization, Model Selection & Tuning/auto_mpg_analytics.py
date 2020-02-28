# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:59:39 2020

@author: Niket
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mpg_df = pd.read_csv('car-mpg.csv')

temp = pd.DataFrame(mpg_df.hp.str.isdigit())
temp[temp['hp'] == False]

mpg_df = mpg_df.replace('?', np.nan)

mpg_df.info()

mpg_df['hp'] = mpg_df['hp'].astype('float64')

car_names = pd.DataFrame(mpg_df[['car_name']])

numeric_cols = mpg_df.drop('car_name', axis = 1)
numeric_cols = numeric_cols.apply(lambda x: x.fillna(x.median()), axis = 0)
mpg_df = numeric_cols.join(car_names)

mpg_df.info()

mpg_df_attr = mpg_df.iloc[:, 0:9]
mpg_df_attr['dispercyl'] = mpg_df_attr['disp'] / mpg_df_attr['cyl']
sns.pairplot(mpg_df_attr, diag_kind='kde')

from scipy.stats import zscore

mpg_df_attr = mpg_df.loc[:, 'mpg': 'origin']
mpg_df_attr_z = mpg_df_attr.apply(zscore)

mpg_df_attr_z.pop('origin')
mpg_df_attr_z.pop('yr')

array = mpg_df_attr_z.values
X = array[: , 1:5]
y = array[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

from sklearn import svm
clr = svm.SVR()
clr.fit(X_train, y_train)

y_pred = clr.predict(X_test)

sns.set(style = "darkgrid", color_codes=True)

with sns.axes_style("white"):
    sns.jointplot(x = y_test, y = y_pred, kind="reg", color="k")
    
mpg_df_attr_z.pop('acc')

array = mpg_df_attr_z.values
X = array[:, 1:5]
y = array[:, 0]
x_train, X_tets, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
clr.fit(X_train, y_train)
y_pred = clr.predict(X_test)

sns.set(style="darkgrid", color_codes=True)

with sns.axes_style("white"):
    sns.jointplot(x=y_test, y = y_pred, kind = "reg", color = "k")
    
cluster_range = range( 2, 6)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters, n_init = 5)
    clusters.fit(mpg_df_attr)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    cluster_errors.append(clusters.inertia_)
clusters_df = pd.DataFrame( {"num_clusters": cluster_range, "cluster_errors": cluster_errors})
clusters_df
    
plt.figure(figsize=(12, 6))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker = 'o')

from sklearn.cluster import KMeans

mpg_df_attr = mpg_df.loc[:, 'mpg': 'origin']
mpg_df_attr_z = mpg_df_attr.apply(zscore)

cluster = KMeans(n_clusters=3, random_state= 2345)
cluster.fit(mpg_df_attr_z)

prediction = cluster.predict(mpg_df_attr_z)
mpg_df_attr_z['GROUP'] = prediction

mpg_df_attr_z_copy = mpg_df_attr_z.copy(deep = True)

centroids = clusters.cluster_centers_
centroids

centroid_df = pd.DataFrame(centroids , columns = list(mpg_df_attr))
centroid_df

import matplotlib.pyplot as plt

mpg_df_attr_z.boxplot(by = "GROUP", layout=(2, 4), figsize = (15, 10))

data = mpg_df_attr_z  
       
def replace(group):
    median, std = group.median(), group.std() 
    outliers = (group - median).abs() > 2*std
    group[outliers] = group.median()       
    return group

data_corrected = (data.groupby('GROUP').transform(replace)) 
concat_data = data_corrected.join(pd.DataFrame(mpg_df_attr_z['GROUP']))

concat_data.boxplot(by = 'GROUP', layout = (2, 4), figsize = (15, 10))

var = 'hp'

with sns.axes_style("white"):
    plot = sns.lmplot(var, 'mpg', data = concat_data, hue = "GROUP")
    
plot.set(ylim = (-3, 3))


var = 'disp'
with sns.axes_style("white"):
    plot = sns.lmplot(var, 'mpg', data = concat_data, hue = "GROUP")
plot.set(ylim = (-3, 3))

var = 'acc'
with sns.axes_style("white"):
    plot = sns.lmplot(var, 'mpg', data = concat_data, hue = "GROUP")

plot.set(ylim = (-3, 3))


var = 'wt'

with sns.axes_style("white"):
    plot = sns.lmplot(var, 'mpg', data= concat_data, hue = "GROUP")
plot.set(ylim = (-3, 3))


largecar = concat_data[concat_data["GROUP"] == 0]
smallcar = concat_data[concat_data["GROUP"] == 1]
sedancar = concat_data[concat_data["GROUP"] == 2]

mpg_df_attr = sedancar.iloc[:, 0:8]
sns.pairplot(mpg_df_attr, diag_kind='kde')

from sklearn import svm
clr = svm.SVR()

array = mpg_df_attr.values
X = array[:, 1:5]
y = array[: , 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
clr.fit(X_train, y_train)
y_pred = clr.predict(X_test)

sns.set(style = "darkgrid", color_codes=True)
with sns.axes_style("white"):
    sns.jointplot(x = y_test, y = y_pred, kind="reg", color = "k")
    
mpg_df_attr_z.boxplot(by = 'origin', layout = (2, 4), figsize = (15, 10))

var = 'acc'

with sns.axes_style("white"):
    plot = sns.lmplot(var, 'mpg', data = mpg_df_attr, hue = 'origin')
    
plot.set(ylim = (0, 50))


mpg_df_attr = mpg_df.loc[:, 'mpg': 'origin']
mpg_df_attr_z = mpg_df_attr.apply(zscore)

cluster = KMeans(n_clusters = 4, random_state =  2345)
cluster.fit(mpg_df_attr_z)

prediction = cluster.predict(mpg_df_attr_z)
mpg_df_attr_z["GROUP"] = prediction

mpg_df_attr_z_copy = mpg_df_attr_z.copy(deep=True)

centroids = cluster.cluster_centers_
centroids

data = mpg_df_attr_z

data_corrected = (data.groupby("GROUP").transform(replace))
concat_data = data_corrected.join(pd.DataFrame(mpg_df_attr_z["GROUP"]))

mpg_df_attr_z.boxplot(by = "GROUP", layout = (2, 4), figsize = (15, 10))

var = 'hp'

with sns.axes_style("white"):
    plot = sns.lmplot(x = var, y = 'mpg', data = concat_data , hue = "GROUP")
plot.set(ylim = (-3, 3))

var = 'disp'
with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))

var = 'wt'
with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=concat_data,hue='GROUP')
plot.set(ylim = (-3,3))

largecar = concat_data[concat_data['GROUP']==1]
smallcar = concat_data[concat_data['GROUP']==0]
sedancar = concat_data[concat_data['GROUP']==2]
minicar  = concat_data[concat_data['GROUP']==3]

mpg_df_attr = sedancar.iloc[:, 0: 8]
sns.pairplot(mpg_df_attr, diag_kind="kde")

clr = svm.SVR()

array = mpg_df_attr.values
X = array[:, 1: 5]
y = array[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

clr.fit(X_train, y_train)

y_pred = clr.predict(X_test)

sns.set(style = "darkgrid", color_codes = True)
with sns.axes_style("white"):
    sns.jointplot(x = y_test, y = y_pred, kind = "reg", color = "k")
    

cols_to_drop = ["cyl", "origin", "GROUP", "acc"]
car_attr = smallcar.drop(cols_to_drop, axis = 1)
car_mpg = np.array(car_attr.pop('mpg'))

from sklearn.decomposition import PCA

cov_matrix = np.cov(car_attr, rowvar = False)
np.linalg.eig(cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

eig_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]

eig_pairs.sort()
eig_pairs.reverse()

eigenvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigenvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

tot = sum(eigenvalues)

var_explained = [(i / tot) for i in sorted(eigenvalues, reverse = True)]

cum_var_exp = np.cumsum(var_explained)

cum_var_exp

plt.bar(range(0, 4), var_explained, alpha = 0.5, align = 'center', label = 'individual explained variance')
plt.step(range(0, 4), cum_var_exp, where = 'mid', label = 'cumulative explained variance')
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal Components")
plt.legend(loc = 'best')

car_mpg = car_mpg.reshape(len(car_mpg), 1)

eigen_space = np.array(eigenvectors_sort[0: 2]).transpose()

names = ['PC1', 'PC2', 'mpg']

proj_data_3D = np.dot(car_attr, eigen_space)

mpg_pca_array = np.concatenate((proj_data_3D, car_mpg), axis = 1)

mpg_pca_df = pd.DataFrame(mpg_pca_array, columns = names)

X = mpg_pca_array[:, 0:1]
y = mpg_pca_array[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

car_mpg.shape

svm.SVR()
clr.fit(X_train, y_train)

y_pred =  clr.predict(X_test)

sns.set(style = "darkgrid", color_codes = True)

with sns.axes_style("white"):
    sns.jointplot(x = y_test, y = y_pred, kind = "reg", color="k")
    
mpg_df_attr = mpg_df.loc[:, 'mpg':'car_type']
mpg_df_attr_z = mpg_df_attr.apply(zscore)
mpg_df_attr_z.boxplot(by = 'car_type',  layout=(2,4), figsize=(15, 10))

var = 'hp'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=mpg_df_attr_z,hue='car_type')
plot.set(ylim = (-3,3))


# mpg Vs wt

var = 'wt'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=mpg_df_attr_z,hue='car_type')
plot.set(ylim = (-3,3))



var = 'disp'

with sns.axes_style("white"):
    plot = sns.lmplot(var,'mpg',data=mpg_df_attr_z,hue='car_type')
plot.set(ylim = (-3,3))

largecar = mpg_df_attr_z[mpg_df_attr_z['car_type'] < 0]   
othercar = mpg_df_attr_z[mpg_df_attr_z['car_type'] > 0]

mpg_df_attr = largecar.iloc[:, 0:9]
sns.pairplot(mpg_df_attr, diag_kind='kde') 

array = mpg_df_attr.values
X = array[:,2:5]
y = array[:,0]   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


clr = svm.SVR()  
clr.fit(X_train , y_train)

y_pred = clr.predict(X_test)

sns.set(style="darkgrid", color_codes=True)

with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");
    
    cols_to_drop = ["cyl", "origin", "acc" , "yr", "car_type"]

car_attr = mpg_df_attr.drop(cols_to_drop , axis = 1)

car_mpg = np.array(car_attr.pop('mpg'))

from sklearn.decomposition import PCA

cov_matrix = np.cov(car_attr, rowvar=False)

np.linalg.eig(cov_matrix)

eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)

eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

eig_pairs.sort()
eig_pairs.reverse()   

eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

tot = sum(eigenvalues)

var_explained = [(i / tot) for i in sorted(eigenvalues, reverse=True)]

cum_var_exp = np.cumsum(var_explained)  

cum_var_exp


car_mpg = car_mpg.reshape(len(car_mpg), 1)

eigen_space = np.array(eigvectors_sort[0:1]).transpose()

proj_data_3D = np.dot(car_attr, eigen_space)



names = ['pc1', 'mpg']

mpg_pca_array = np.concatenate((proj_data_3D, car_mpg), axis=1)

mpg_pca_df = pd.DataFrame(mpg_pca_array ,columns=names )


X = mpg_pca_array[:,0:1] 
y = mpg_pca_array[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

clr = svm.SVR()  
clr.fit(X_train , y_train)

y_pred = clr.predict(X_test)
          
with sns.axes_style("white"):
    sns.jointplot(x=y_test, y=y_pred, kind="reg", color="k");