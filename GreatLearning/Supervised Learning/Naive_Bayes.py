# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:45:39 2019

@author: N827941
"""

# We will use Naive Bayes to model the "Pima Indians Diabetes" data set. This model will predict which people are likely to develop diabetes.

#This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# load and review the data
pdata = pd.read_csv('pima-indians-diabetes.csv')
pdata.shape

pdata.head()

pdata.isnull().values.any() # check is there any missing values
columns = list(pdata)[0:-1]
pdata[columns].hist(stacked = False, bins = 100, figsize = (12, 30))
# Histgram for first 8 columns

corr_matrix = pdata.corr() # find the correlation matrix

# However we want to see correlation in graphical representation so below is function for that
def plot_corr(df, size = 11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr(pdata)

sns.pairplot(pdata, diag_kind = 'kde')

n_true = len(pdata[pdata['class'] == True])
n_false = len(pdata[pdata['class'] == False])
print('Number of true cases:{0} ({1:2.2f}%)'.format(n_true, (n_true / (n_true + n_false)) * 100))
print('Number of false cases:{0} ({1:2.2f}%)'.format(n_false, (n_false / (n_true + n_false)) * 100))

# So we have 34.90% people in current data set who have diabetes and rest of 65.10% doesn't have diabetes. 
# Its a good distribution True/False cases of diabetes in data.

# Splitting the data
# will use 70% data for training and 30% for the test
from sklearn.model_selection import train_test_split

X = pdata.drop('class', axis = 1)
y = pdata['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

X_train.head()

# lets check the split of the data
print('({0:0.2f}%) of the train data'.format((len(X_train) / len(pdata.index)) * 100)) 
print('({0:0.2f}%) of the test data'.format((len(X_test) / len(pdata.index)) * 100))

# Now lets check diabetes True/False ratio in split data 
print("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(pdata.loc[pdata['class'] == 1]), (len(pdata.loc[pdata['class'] == 1])/len(pdata.index)) * 100))
print("Original Diabetes False Values   : {0} ({1:0.2f}%)".format(len(pdata.loc[pdata['class'] == 0]), (len(pdata.loc[pdata['class'] == 0])/len(pdata.index)) * 100))
print("")
print("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))
print("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))
print("")
print("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))
print("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))
print("")

# Data Preparation
# Check hidden missing values
X_train.head()

# replace the missing values with mean values
from sklearn.impute import SimpleImputer
rep_0 = SimpleImputer(missing_values = 0, strategy = "mean")
cols = X_train.columns
X_train = pd.DataFrame(rep_0.fit_transform(X_train))
X_test = pd.DataFrame(rep_0.fit_transform(X_test))

X_train.columns = cols
X_test.columns = cols

# train Naive bayes 
from sklearn.naive_bayes import GaussianNB
diab_model = GaussianNB()
diab_model.fit(X_train, y_train)

# Performance of our model with training data
diab_model_predict = diab_model.predict(X_train)
from sklearn import metrics

print('Model Accuracy {0:.4f}%'.format(metrics.accuracy_score(y_train, diab_model_predict)))

# Performance of our model with test data
diab_model_predict = diab_model.predict(X_test)
print('Model Accuracy {0:.4f}%'.format(metrics.accuracy_score(y_test, diab_model_predict)))

# Lets check the confusion matrix and classification report
print('Confusion Matrix')
cm = metrics.confusion_matrix(y_test, diab_model_predict, labels = [1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1", "0"]], 
                         columns = [i for i in ["Predict 1", "Predict 0"]])

# plot the confustion matrix using heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df_cm, annot = True)

print("Classification Report")
print(metrics.classification_report(y_test, diab_model_predict, labels=[1, 0]))

