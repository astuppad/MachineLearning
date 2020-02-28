# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:58:03 2020

@author: Niket
"""

# import the libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the data from the dataset
cData = pd.read_csv("auto-mpg.csv")
cData.shape

cData.head()

cData = cData.drop(['car name', 'origin'], axis = 1)



# Dealing with missing data
# isdigit()? on 'horsepower' 
hpIsDigit = pd.DataFrame(cData.hp.str.isdigit())

#print isDigit = False!
cData[hpIsDigit['hp'] == False]

# There are various ways to handle missing values. 
# Drop the rows, replace missing values with median values etc. 
# of the 398 rows 6 have NAN in the hp column. We could drop those 6 rows - 
# which might not be a good idea under all situations. Here, we will replace them 
# with their median values. First replace '?' with NaN and then replace NaN with median

cData = cData.replace('?', np.nan)
cData[hpIsDigit['hp'] == False]

#instead of dropping the rows, lets replace the missing values with median value. 
cData.median()

# replace the missing values with median value.
# Note, we do not need to specify the column names below
# every column's missing value is replaced with that column's median respectively  (axis =0 means columnwise)
#cData = cData.fillna(cData.median())
medianFiller = lambda x: x.fillna(x.median())
cData = cData.apply(medianFiller, axis = 0)

cData['hp'] = cData['hp'].astype('float64')

# independant variables
X = cData.drop(['mpg'], axis=1)
# the dependent variable
y = cData[['mpg']]

sns.pairplot(X, diag_kind='kde')   # to plot density curve instead of histogram on the diag

from scipy.stats import zscore
Xscaled = X.apply(zscore)

covMatrix = np.cov(Xscaled, rowvar = False)
print(covMatrix)

pca = PCA(n_components = 6)
pca.fit(Xscaled)

# Eigen Values
print(pca.explained_variance_)

#Eigen Vectors 
print(pca.components_)

# And the percentage of variation explained by each eigen Vector
print(pca.explained_variance_ratio_)

# bar plot for explained variance ratio
plt.bar(list(range(1, 7)), pca.explained_variance_ratio_, alpha = 0.5, align = 'center')
plt.ylabel('Vairation explained')
plt.xlabel('eigen value')

# cumilative plot for exlpaine variance ratio
plt.step(list(range(1, 7)), np.cumsum(pca.explained_variance_ratio_), where = 'mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('eigen Value')

# Dimensionality reduction
# Now 3 dimensions seems very reasonable. With 3 variables we can explain over 95% of the variation in the original data!
pca3 = PCA(n_components=3)
pca3.fit(Xscaled)
print(pca3.components_)
print(pca3.explained_variance_ratio_)
Xpca3 = pca3.transform(Xscaled)

Xpca3

sns.pairplot(pd.DataFrame(Xpca3))

# Fit the linear Model
# Lets construct two linear models. The first with all the 6 independent variables and the second with only the 3 new variables constructed using PCA.
regression_model = LinearRegression()
regression_model.fit(Xscaled, y)
regression_model.score(Xscaled, y)

regression_model_pca = LinearRegression()
regression_model_pca.fit(Xpca3, y)
regression_model_pca.score(Xpca3, y)

# Looks like by drop reducing dimensionality by 3, we only dropped around 3% in R^2! This is insample (on training data) and hence a drop in R^2 is expected. Still seems easy to justify the dropping of variables. An out of sample (on test data), with the 3 independent variables is likely to do better since that would be less of an over-fit. 
