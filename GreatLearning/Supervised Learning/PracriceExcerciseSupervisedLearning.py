# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:56:03 2019

@author: N827941
"""
# 1. Load Libraries and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
from sklearn import metrics
from sklearn import datasets
import seaborn as sns

credit_df = pd.read_excel('German_Credit.xlsx')

credit_df.head()

# 2. Check how many records do we have
credit_df.shape

# 3. Plot Histogram for column 'CreditAmount'
plt.hist(credit_df['CreditAmount'])


amountIntervalsPoint = np.array([0, 500, 1000, 1500, 2000, 2500, 5000, 7500, 15000, 20000])
amountIntervals = [(amountIntervalsPoint[i] + int( i != 0), amountIntervalsPoint[i + 1]) for i in np.arange(len(amountIntervalsPoint) - 1)]
amountIntervals 

amountIntervalsDf = pd.DataFrame(amountIntervals, columns = ['intervalLeftSide', 'intervalRightSide'])
amountIntervalsDf

Credibility0 = []
Credibility1 = []
for interval in amountIntervals:
    subData = credit_df[credit_df.CreditAmount >= interval[0]]
    subData = credit_df[credit_df.CreditAmount <= interval[1]]
    Credibility0.append(sum(subData.Creditability == 0))
    Credibility1.append(sum(subData.Creditability == 1))
    
tempDf = pd.DataFrame(np.column_stack([Credibility0, Credibility1]), columns = ['Credibility0', 'Credibility1'])
tempDf

# 4. Concatenate the above 2 dataframes and give the total of Credibiliity0 and Credibiliity1¶
compareCreditWorthinessDf = pd.concat([amountIntervalsDf.reset_index(drop = True), tempDf], axis = 1)
compareCreditWorthinessDf['total'] = compareCreditWorthinessDf.Credibility0 + compareCreditWorthinessDf.Credibility1

# 5. Plot Creditworthiness plot for Credibility == 0 and also ==1
plt.plot(compareCreditWorthinessDf.Credibility0)
plt.xlabel('credit amount interval number')
plt.ylabel('Probabilty')
plt.title("Creditworthiness plot for Credibility == 0")

plt.plot(compareCreditWorthinessDf.Credibility1)
plt.xlabel('credit amount interval number')
plt.ylabel(' Probability')
plt.title("Credit worthiness plot for Credibility == 1")

# 6. Prepare input data for the model
X = np.array(credit_df.CreditAmount)
y = credit_df.Creditability.astype('category')

# 7. Fit logistic regression model¶
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
logit = sm.Logit(y_train, sm.add_constant(X_train))
lg = logit.fit()
lg.summary2()

# 8. Test accuracy calculation
def get_predictions( y_test, model):
    y_pred_df = pd.DataFrame( {'actual' : y_test,
                               'predicted_prob': lg.predict(sm.add_constant(X_test))})
    return y_pred_df

X_test[0:5]
y_pred_df = get_predictions(X_test, lg)
y_pred_df['originalCredibility'] = np.array(y_test)
    
y_pred_df['predicted'] = y_pred_df.predicted_prob.map(lambda x: 1 if x > 0.6 else 0)
y_pred_df[0 : 10]

# 9. Build a confusion matrix
def draw_cm(actual, predicted):
    cm = metrics.confusion_matrix(actual, predicted, [1, 0])
    sns.heatmap(cm, annot = True, fmt = '.2f', xticklabels = ['Default', 'No Default'], yticklabels = ['Default', 'No Default'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
draw_cm(y_pred_df.originalCredibility, y_pred_df.predicted)

# 10. Predicted Probability distribution Plots for Defaults and Non Defaults
sns.distplot( y_pred_df[y_pred_df.originalCredibility == 1]['predicted_prob'], kde = False, color = 'b')
sns.distplot( y_pred_df[y_pred_df.originalCredibility == 0]['predicted_prob'], kde = False, color = 'g')

