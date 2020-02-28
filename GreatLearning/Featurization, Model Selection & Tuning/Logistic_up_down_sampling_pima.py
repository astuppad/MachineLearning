# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 08:21:49 2020

@author: Niket
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import numpy as np

from sklearn import metrics
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE

pima_df = pd.read_csv('pima-indians-diabetes.csv')
pima_df.head()

pima_df[~pima_df.applymap(np.isreal).all(1)]

pima_df.groupby(["class"]).count()

pima_df_attr = pima_df.iloc[:, 0:9]
plt.tight_layout()
plt.savefig('pima_paipanel.png')

sns.pairplot(pima_df, hue = "class", diag_kind = "kde")

array = pima_df.values
X = array[:, 0:7]
y = array[:, 8]
test_size = 0.30
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = seed)
type(X_train)

# SMOTE
print("Before Upsampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before Upsampling, counts of label '0': {}".format(sum(y_train == 0)))

sm = SMOTE(sampling_strategy= 1, k_neighbors=5, random_state = 1)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print("After Upsampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After Upsampling, counts of label '0': {}".format(sum(y_train_res == 0)))

print('After UpSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After UpSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)

test_pred = model.predict(X_test)
print(metrics.classification_report(y_test, test_pred))
print(metrics.confusion_matrix(y_test, test_pred))

# Upsampling smaller class
model.fit(X_train_res, y_train_res)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.classification_report(y_test, y_predict))
print(metrics.confusion_matrix(y_test, y_predict))


# Downsampling the larger class
non_diab_indices = pima_df[pima_df["class"] == 0].index
no_diab = len(pima_df[pima_df["class"] == 0])
print(no_diab)

diab_indices = pima_df[pima_df["class"]==1].index
diab = len(pima_df[pima_df["class"]==1])
print(diab)

random_indices = np.random.choice(non_diab_indices, no_diab - 200, replace = False)

down_sample_indices = np.concatenate([diab_indices, random_indices])

pima_df_down_sample = pima_df.loc[down_sample_indices]
pima_df_down_sample.shape
pima_df_down_sample.groupby(["class"]).count()

array = pima_df_down_sample.values
X = array[:, 0:7]
y = array[:, 8]
test_size = 0.30
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state = seed)

print('After DownSampling, the shape of X_train: {}'.format(X_train.shape))
print('After DownSampling, the shape of X_test: {} \n'.format(X_test.shape))

model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.classification_report(y_test, y_predict))
print(metrics.confusion_matrix(y_test, y_predict))

# IMBLearn Random Under Sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state = seed)

X_rus, y_rus = rus.fit_sample(X_train, y_train)

y_rus

y_rus.shape
len(y_rus == 1)
len(y_rus == 0)

# Imblearn random over sampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state = seed)

X_ros, y_ros = ros.fit_sample(X_train, y_train)

y_ros

y_ros.shape

X_ros.shape

# Tomeklinks
from imblearn.under_sampling import TomekLinks
tl = TomekLinks()
X_tl, y_tl = tl.fit_sample(X_train, y_train)

y_tl.shape
X_tl.shape

# Upsampling followed by down sampling
from imblearn.combine import SMOTETomek
smt = SMOTETomek()

X_smt, y_smt = smt.fit_sample(X_train, y_train)

X_smt.shape
y_smt.shape

# Cluster based under sampling
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids()
X_cc, y_cc = cc.fit_sample(X_train, y_train)

X_cc.shape
y_cc.shape
