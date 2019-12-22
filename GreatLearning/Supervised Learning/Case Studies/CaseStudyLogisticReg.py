# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 08:58:30 2019

@author: N827941
"""
# Import pandas and linear Model
import pandas as pd
from sklearn.linear_model import LogisticRegression

# importing ploting libraries
import matplotlib.pyplot as plt

#Let us break the X and y dataframes into training set and test set. For this we will use
#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split

# calculate accuracy measures and confusion matrix
from sklearn import metrics

df = pd.read_csv('CreditRisk.csv')
df.head()

cr_df = df.drop('Loan_ID', axis = 1)
cr_df.head()
cr_df['Loan_Amount_Term'].value_counts(normalize = True)

# The Loan_Amount_Term is highly skewed - so we will delete this column
cr_df.drop('Loan_Amount_Term', axis = 1, inplace = True)

# every column's missing value is replaced with 0 respectively
cr_df = cr_df.fillna('0')
cr_df.head()

#Lets analysze the distribution of the various attribute
cr_df.describe().transpose()

# Let us look at the target column which is 'Loan_Status' to understand how the data is distributed amongst the various values
cr_df.groupby(['Loan_Status']).mean()

# Convert X & Y variable to a categorical variable as relevant
cr_df['Loan_Status'] = cr_df['Loan_Status'].astype('category')
cr_df['Credit_History'] = cr_df['Credit_History'].astype('category')

cr_df.info()

# Calculate baseline proportion - ratio of Yes to No to identify data imbalance
prop_Y = cr_df['Loan_Status'].value_counts(normalize = True)
print(prop_Y)

# Model building
## Define X and Y variables
X = cr_df.drop('Loan_Status', axis = 1)
y = cr_df['Loan_Status']

# convert categorical variables to dummy variables
X = pd.get_dummies(X, drop_first = True)
##Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# build the Logistic regression model
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

# Use score method to get accuracy of model
score = logisticRegr.score(X_test, y_test)

#Interpretation of Pseudo R^2
#A pseudo R^2 of 29% indicates that 29% of the uncertainty of the intercept only model is explained by the full model
#
#Calculate the odds ratio from the coef using the formula odds ratio=exp(coef)
#Calculate the probability from the odds ratio using the formula probability = odds / (1+odds)

#gcoef = pd.DataFrame(lg.params, columns=['coef'])
#lgcoef.loc[:, "Odds_ratio"] = np.exp(lgcoef.coef)
#lgcoef['probability'] = lgcoef['Odds_ratio']/(1+lgcoef['Odds_ratio'])
#lgcoef['pval']=lg.pvalues
#pd.options.display.float_format = '{:.2f}'.format
#
## FIlter by significant p-value (pval <0.1) and sort descending by Odds ratio
#lgcoef = lgcoef.sort_values(by="Odds_ratio", ascending=False)
#pval_filter = lgcoef['pval']<=0.1
#lgcoef[pval_filter]
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Predict for train set
pred_train = logreg.predict(X_train)

mat_train = metrics.confusion_matrix(y_train, pred_train)

print('Confusion matrix\n', mat_train)

# Predict for test set
pred_test = logreg.predict(X_test)

mat_test = metrics.confusion_matrix(y_test, pred_test)
print('Confusion matrix\n', mat_test)

logit_roc_auc = metrics.roc_auc_score(y_test, pred_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label = 'Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0,1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Reciever Operating Charecteristic')
plt.legend(loc = 'lower right')


