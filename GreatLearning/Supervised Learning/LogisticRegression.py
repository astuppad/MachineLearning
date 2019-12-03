# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 08:25:22 2019

@author: N827941
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pdata = pd.read_csv('pima-indians-diabetes.csv')

pdata.shape

pdata.head()

pdata.isnull().values.any()

columns = list(pdata)[0 : -1]
pdata[columns].hist(stacked = True, bins = 100, figsize = (12, 30), layout = (14, 2))

pdata.corr()

def plot_corr(df, size = 11):
    corr = df.corr()
    fig , ax = plt.subplots(figsize = (size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
plot_corr(pdata)

sns.pairplot(pdata, diag_kind = 'kde')

# Calculate diabetes ratio True/False from outcome variable
n_true = len(pdata.loc[pdata['class'] == True])
n_false = len(pdata.loc[pdata['class'] == False])
print('Number of true cases: {0} ({1:2.2f}%)'.format(n_true, (n_true /  (n_true + n_false)) * 100))
print('Number of true cases: {0} ({1:2.2f}%)'.format(n_false, (n_false / (n_false + n_true)) * 100))

# Splitting the data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = pdata.drop('class', axis = 1)
y = pdata['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

X_train.head()

# Lets check the split of the data.
print("{0: 0.2f}% data is in training set".format((len(X_train)/len(pdata.index)) * 100))
print("{0: 0.2f}% data is in test set".format((len(X_test)/len(pdata.index)) * 100))

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


X_train.head()


#from sklearn.preprocessing import Imputer
#my_imputer = Imputer()
#data_with_imputed_values = my_imputer.fit_transform(original_data)
from sklearn.impute import SimpleImputer
rep_0 = SimpleImputer(missing_values=0, strategy="mean")
cols=X_train.columns
X_train = pd.DataFrame(rep_0.fit_transform(X_train))
X_test = pd.DataFrame(rep_0.fit_transform(X_test))

X_train.columns = cols
X_test.columns = cols

X_train.head()

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Fit the model on train
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)
#predict on test
y_predict = model.predict(X_test)


coef_df = pd.DataFrame(model.coef_)
coef_df['intercept'] = model.intercept_
print(coef_df)

model_score = model.score(X_test, y_test)
print(model_score)


cm = metrics.confusion_matrix(y_test, y_predict, labels = [1, 0])
df_cm = pd.DataFrame(cm, index = [i for i in ["1", "0"]], 
                     columns = [i for i in ["Predict 1", "Predict 0"]])

plt.figure(figsize = (7, 5))
sns.heatmap(df_cm, annot = True)
