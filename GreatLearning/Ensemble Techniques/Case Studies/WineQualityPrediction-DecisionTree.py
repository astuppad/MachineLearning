# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:49:15 2020

@author: Niket
"""

"""

Case Study - Wine Quality Prediction

Context
This datasets is related to red variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones).

Dataset:
https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
This dataset is also available from the UCI machine learning repository, https://archive.ics.uci.edu/ml/datasets/wine+quality 

Problem Statement:
Wine Quality Prediction- Here, we will apply a method of assessing wine quality using a decision tree, and test it against the wine-quality dataset from the UC Irvine Machine Learning Repository. The wine dataset is a classic and very easy multi-class classification dataset.

Number of Instances: red wine - 1599; white wine - 4898.
Attribute information:
Input variables (based on physicochemical tests):
fixed acidity (tartaric acid - g / dm^3)
volatile acidity (acetic acid - g / dm^3)
citric acid (g / dm^3)
residual sugar (g / dm^3)
chlorides (sodium chloride - g / dm^3
free sulfur dioxide (mg / dm^3)
total sulfur dioxide (mg / dm^3)
density (g / cm^3)
pH
sulphates (potassium sulphate - g / dm3)
alcohol (% by volume)
Output variable (based on sensory data): 
quality (score between 0 and 10)
Missing Attribute Values: None

Description of attributes:
1 - fixed acidity: most acids involved with wine or fixed or nonvolatile (do not evaporate readily)
2 - volatile acidity: the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
3 - citric acid: found in small quantities, citric acid can add 'freshness' and flavor to wines
4 - residual sugar: the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet
5 - chlorides: the amount of salt in the wine
6 - free sulfur dioxide: the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine
7 - total sulfur dioxide: amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
8 - density: the density of water is close to that of water depending on the percent alcohol and sugar content
9 - pH: describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
10 - sulphates: a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant
11 - alcohol: the percent alcohol content of the wine
Output variable (based on sensory data): 12 - quality (score between 0 and 10)
"""

# import the libraries and load the data
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer #DT does not take strings as input for the model fit step....

wine_df = pd.read_csv('winequality-red.csv', sep = ';')

# Print 10 samples from the dataset
wine_df.head(10)

# Print the datatypes of each column and the shape of the dataset
wine_df.dtypes

wine_df.shape

# Print the descriptive statistics of each & every column using describe() function
wine_df.describe().transpose()

# Using univariate analysis check the individual attributes for their basic statistic such as central values, spread, tails etc.

# plot the graphsof different variable to see the distribution.
sns.countplot(wine_df['quality'])

sns.distplot(wine_df['fixed acidity'])

# Use correlation method to observe the relationship between different variables and state your insights.
plt.figure(figsize = (10, 8))
sns.heatmap(wine_df.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            cmap="YlGnBu")

# Levels of y variable
wine_df['quality'].value_counts()

# Combine 7&8 together; combine 3 and 4 with 5 so that we have only 3 levels and more balanced y variable
wine_df['quality'] = wine_df['quality'].replace(8,7)
wine_df['quality'] = wine_df['quality'].replace(3,4)
wine_df['quality'] = wine_df['quality'].replace(4,5)
wine_df['quality'].value_counts()

# Split the wine_df into training and test set in the ratio of 70:30 (Training:Test) based on dependent and independent variables.
X_train, X_test, y_train, y_test = train_test_split(wine_df.drop('quality', axis = 1), 
                                                    wine_df['quality'], test_size = 0.3, random_state = 1)


# Create the decision tree model using “entropy” method of finding the split columns and fit it to training data.
model_entropy = DecisionTreeClassifier(criterion='entropy')
model_entropy.fit(X_train, y_train)

print(model_entropy.score(X_train, y_train)) # Performance on the training data

print(model_entropy.score(X_test, y_test)) # Performance on test data.

# There is a high degree of overfitting in the model due to which the test accuracy drops drastically. This shows why decision trees are prone to overfitting.

# Regularize/prune the decision tree by limiting the max. depth of trees and print the accuracy.

clf_pruned = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
clf_pruned.fit(X_train, y_train)

print(clf_pruned.score(X_train, y_train))

print(clf_pruned.score(X_test, y_test))

# Visualizing the tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import graphviz

xvar = wine_df.drop('quality', axis = 1)
feature_cols = xvar.columns

dot_data = StringIO()
export_graphviz(clf_pruned, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1', '2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('wines_pruned.png')
Image(graph.create_png())

preds_pruned = clf_pruned.predict(X_test)
preds_pruned_train = clf_pruned.predict(X_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, preds_pruned))
print(accuracy_score(y_train, preds_pruned_train))
acc_DT = accuracy_score(y_test, preds_pruned)

# When the tree is regularised, overfitting is reduced, but there is no increase in accuracy

feature_importance = clf_pruned.tree_.compute_feature_importances(normalize = False)

feat_imp_dict = dict(zip(feature_cols, clf_pruned.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient = 'index')
feat_imp.sort_values(by=0, ascending = False)

# From the feature importance dataframe we can infer that alcohol, sulphate, volatile acidity and total sulfur dioxide are the variables that impact wine quality
#Store the accuracy results for each model in a dataframe for final comparison
resultsDf = pd.DataFrame({'Method': ['Decision Tree'], 'accuracy': acc_DT})
resultsDf = resultsDf[['Method', 'accuracy']]
resultsDf.head()
