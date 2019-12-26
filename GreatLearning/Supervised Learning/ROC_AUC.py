# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 19:18:27 2019

@author: N827941
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

import pylab as pl
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
random_state = np.random.RandomState(0)

diabetes = read_csv('pima-indians-diabetes-2.csv')

cols = diabetes.columns
cols
array = diabetes.values
X = array[:, 0:8]
y = array[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 1)
lgClassifier = LogisticRegression()
svClassifier = svm.SVC(kernel = 'linear', probability = True, random_state = 1)
probalg = lgClassifier.fit(X_train, y_train).predict_proba(X_test)
probasv = svClassifier.fit(X_train, y_train).predict_proba(X_test)

fpr1, tpr1, threshold1 = roc_curve(y_test, probalg[:, 1])
roc_auc1 = auc(fpr1, tpr1)
print('Area under the curve is {0}', roc_auc1)

fpr2, tpr2, threshold2 = roc_curve(y_test, probasv[:, 1])
roc_auc2 = auc(fpr2, tpr2)
print('Area under the curve is {0}', roc_auc2)

pl.clf()
pl.plot(fpr1, tpr1, label = 'ROC curve for logistic (area = %0.2f)' % roc_auc1)
pl.plot(fpr2, tpr2, label = 'ROC curve for SVC (area = %0.2f)' % roc_auc2)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Reciever Operatice Charecterists example')
pl.legend(loc = 'lower right')

import pandas as pd
i = np.arange(len(tpr1))
roc1 = pd.DataFrame({'fpr1': pd.Series(fpr1, index = i), 'tpr1': pd.Series(tpr1, index = i), '1-fpr1': pd.Series(1-fpr1, index = i), 'thesholds1': pd.Series(threshold1, index = i)})

i = np.arange(len(tpr2))
roc2 = pd.DataFrame({'fpr2': pd.Series(fpr2, index = i), 'tpr2': pd.Series(tpr2, index = i), '1-fpr2': pd.Series(1-fpr2, index = i), 'thesholds2': pd.Series(threshold2, index = i)})
