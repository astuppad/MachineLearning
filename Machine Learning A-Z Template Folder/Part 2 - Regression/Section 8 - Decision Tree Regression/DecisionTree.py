# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:40:13 2019

@author: Niket
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting the Dicision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predict the new result
y_pred = regressor.predict([[6.5]])

# Visualising the Regression result
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff(Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salaries')
plt.show()

# Visualising the regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth Or Bluff(Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salaries')
plt.show()


