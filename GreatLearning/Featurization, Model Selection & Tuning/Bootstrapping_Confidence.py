# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 08:40:19 2020

@author: Niket
"""

from pandas import read_csv
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import numpy as np

data = read_csv('pima-indians-diabetes.csv')
values = data.values

n_iterations = 100

stats = list()
for i in range(n_iterations):
    train = resample(values)
    test = np.array([x for x in values if x.tolist() not in train.tolist()])
    model = DecisionTreeClassifier()
    model.fit(train[:, :-1], train[:, -1])
    predictions = model.predict(test[:, :-1])
    score = accuracy_score(test[:, -1], predictions)
    print(score)
    stats.append(score)
    
pyplot.hist(stats)

alpha = 0.99
p = ((1.0 - alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha + ((1.0 - alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
