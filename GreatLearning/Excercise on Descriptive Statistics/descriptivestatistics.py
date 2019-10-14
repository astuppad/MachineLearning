# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:55:27 2019

@author: N827941
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

mydata = pd.read_csv('CardioGoodFitness-1.csv')
mydata.describe(include = 'all')
mydata.info()
mydata.hist(figsize = (20, 30))
sns.boxplot(x="Gender", y="Age", data=mydata)
pd.crosstab(mydata['Product'], mydata['Gender'])
pd.crosstab(mydata['Product'], mydata['MaritalStatus'])
sns.countplot(x="Product", hue="Gender", data=mydata)
pd.pivot_table(mydata, index=['Product','Gender'],
                columns='MaritalStatus', aggfunc=len)
pd.pivot_table(mydata, 'Income', index=['Product', 'Gender'],
               columns='MaritalStatus')
pd.pivot_table(mydata, 'Miles', index = ['Product', 'Gender'],
               columns='MaritalStatus')
sns.pairplot(mydata)
mydata['Age'].std()
mydata['Age'].mean()
sns.distplot(mydata['Age'])
mydata.hist(by='Gender', column='Income')
mydata.hist(by='Gender', column='Miles')
cov = mydata.cov()
corr = mydata.corr()
sns.heatmap(corr, annot = True)

from sklearn import linear_model
regr = linear_model.LinearRegression()

x = mydata['Miles']
y= mydata[['Usage', 'Fitness']]

regr.fit(x, y)

regr.coef_
regr.intercept_