# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:21:50 2019

@author: N827941
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn import tree
from os import system

creditData = pd.read_csv('credit.csv')
creditData.head(10)

creditData.shape

creditData.describe()

creditData.info()  # many columns are of type object i.e. strings. These need to be converted to ordinal type

# Lets convert the columns with an 'object' datatype into categorical variables
for feature in creditData.columns:
    if(creditData[feature].dtype == 'object'):
        creditData[feature] = pd.Categorical(creditData[feature])

creditData.info()

print(creditData.checking_balance.value_counts())
print(creditData.credit_history.value_counts())
print(creditData.purpose.value_counts())
print(creditData.savings_balance.value_counts())
print(creditData.employment_duration.value_counts())
print(creditData.other_credit.value_counts())
print(creditData.housing.value_counts())
print(creditData.job.value_counts())
print(creditData.phone.value_counts())
print(creditData.default.value_counts())

replaceStruct = {
            "checking_balance": {"unknown" : 1, "< 0 DM": 2, "1 - 200 DM": 3, "> 200 DM": 4},
            "credit_history": {"critical": 1, "poor": 1, "good": 3, "very good": 4, "perfect": 5},
            "savings_balance": {"unknown": -1, "< 100 DM": 1, "100 - 500 DM": 2, "500 - 1000 DM": 3, "> 1000 DM": 4},
            "employment_duration": {"unemployed": 1, "< 1 year": 2, "1 - 4 years": 2, "4 - 7 years": 3, "> 7 years": 4},
            "phone": {"no": 0, "yes": 1},
            "default": {"no": 0, "yes": 1}
        }

oneHotCols = ["purpose", "housing", "other_credit", "job"]

creditData = creditData.replace(replaceStruct)
creditData = pd.get_dummies(creditData, columns = oneHotCols)

creditData.info()

# Split Data
X = creditData.drop("default", axis = 1)
y = creditData.pop("default")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 1)

# build decision tree model
dTree = DecisionTreeClassifier(criterion = 'gini', random_state = 1)
dTree.fit(X_train, y_train)

# scoring our decision
print(dTree.score(X_train, y_train))
print(dTree.score(X_test, y_test))

# visualizing the decision tree
import graphviz 
dot_data = tree.export_graphviz(dTree, out_file = None,
                                feature_names = list(X_train),
                                class_names = ["No", "Yes"],
                                filled = True,
                                rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data)
graph

#tree.export_graphviz outputs a .dot file. This is a text file that describes a graph structure using a specific structure. You can plot this by

#pasting the contents of that file at http://webgraphviz.com/ (or)
#generate a image file using the 'dot' command (this will only work if you have graphviz installed on your machine)

#Works only if "dot" command works on you machine
retCode = system("dot -Tpng credit_tree.dot -o credit_tree.png")
if(retCode>0):
    print("system command returning error: "+str(retCode))
else:
    display(Image("credit_tree.png"))
    
# Reducing over fitting (regularizing)
dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state=1)
dTreeR.fit(X_train, y_train)
print(dTreeR.score(X_train, y_train))
print(dTreeR.score(X_test, y_test))

train_char_label = ['No', 'Yes']
Credit_Tree_FileR = open('credit_treeR.dot','w')
dot_data = tree.export_graphviz(dTreeR, out_file=Credit_Tree_FileR, feature_names = list(X_train), class_names = list(train_char_label))
Credit_Tree_FileR.close()

#Works only if "dot" command works on you machine

retCode = system("dot -Tpng credit_treeR.dot -o credit_treeR.png")
if(retCode>0):
    print("system command returning error: "+str(retCode))
else:
    display(Image("credit_treeR.png"))
    
# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )
print(pd.DataFrame(dTreeR.feature_importances_, columns = ["Imp"], index = X_train.columns))

y_pred = dTreeR.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred, labels = [1, 0])

df_cm = pd.DataFrame(cm, columns = ["Yes", "No"],
                         index = ["Yes", "No"])

plt.figure(figsize = (7, 5))
sns.heatmap(df_cm, annot = True, fmt = 'g')

#Ensemble Learning - Bagging
from sklearn.ensemble import BaggingClassifier
bgcl = BaggingClassifier(base_estimator = dTree, n_estimators = 50, random_state = 1)
bgcl.fit(X_train, y_train)

y_predict = bgcl.predict(X_test)

print(bgcl.score(X_test, y_test))
cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')

#Ensemble Learning - AdaBoosting
from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(n_estimators=10, random_state=1)
#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)
abcl = abcl.fit(X_train, y_train)

y_predict = abcl.predict(X_test)
print(abcl.score(X_test , y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')

# Ensemble Learning - GradientBoost
from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 50, random_state = 1)
gbcl.fit(X_train, y_train)

y_predict = gbcl.predict(X_test)
print(gbcl.score(X_test , y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')

# Ensemble RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators = 50, random_state = 1, max_features = 12)
rfcl.fit(X_train, y_train)

y_predict = rfcl.predict(X_test)
print(rfcl.score(X_test, y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')
