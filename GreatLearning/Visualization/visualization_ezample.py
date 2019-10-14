# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:33:44 2019

@author: N827941
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set(color_codes=True)


auto = pd.read_csv('Automobile_data.csv')
auto.head()
auto['normalized-losses'] = auto['normalized-losses'].replace('?', np.nan)
auto['normalized-losses'] =auto['normalized-losses'].astype('float64')
auto['normalized-losses'] = auto.fillna(value = auto['normalized-losses'].mean())
sns.distplot(auto['normalized-losses'])
plt.show()

# single/uni variant distribution
sns.distplot(auto['wheel-base'], kde = False, rug = True) # No kernel density but rug is True
sns.jointplot(auto['engine-size'], auto['horsepower'])


auto['engine-size'] =auto['engine-size'].astype('int')
auto['horsepower'] = auto['normalized-losses'].replace('?', np.nan)
auto['horsepower'] = auto['horsepower'].astype('int')
# bi variant distribution plot
sns.jointplot(auto['engine-size'], auto['horsepower'], kind='hex')

sns.jointplot(auto['engine-size'], auto['horsepower'], kind='reg') # reg is regression line
sns.jointplot(auto['engine-size'], auto['horsepower'], kind='kde') # kde is kernel density

# multi variant distribution
# mutliple variable single distribution
sns.pairplot(auto[['normalized-losses', 'engine-size', 'horsepower']])

# plot 2 continous variables 
# one with catogorical and one with continous
sns.stripplot(auto['fuel-type'], auto['horsepower'])
sns.stripplot(auto['fuel-type'], auto['horsepower'], jitter = True) # distribution of points
sns.swarmplot(auto['fuel-type'], auto['horsepower']) # use for no overlapping of points
sns.boxplot(auto['num-of-doors'], auto['horsepower'])
sns.boxplot(auto['num-of-doors'], auto['horsepower'], hue = auto['fuel-type'])

sns.barplot(auto['body-style'], auto['horsepower']) #shows the confidence level also
sns.barplot(auto['body-style'], auto['horsepower'], hue = auto['engine-location'])

sns.countplot(auto['body-style']) # shows the count based body-style
sns.countplot(auto['body-style'], hue= auto['engine-location'])

sns.pointplot(auto['fuel-system'], auto['horsepower']) # line graph connect continous values
sns.pointplot(auto['fuel-system'], auto['horsepower'], hue = auto['num-of-doors'])

# mutli panel categorical plots
sns.factorplot(x = 'fuel-type',
               y = 'horsepower',
               hue = 'num-of-doors',
               col = 'engine-location',
               data = auto,
               kind = 'swarm')  # no overlapping of point in swarm

# various types of kind input : (point, bar, count, box, voilin strip)
sns.factorplot(x = 'fuel-type',
               y = 'horsepower',
               hue = 'num-of-doors',
               col = 'engine-location',
               data = auto,
               kind = 'point')

sns.factorplot(x = 'fuel-type',
               y = 'horsepower',
               hue = 'num-of-doors',
               col = 'engine-location',
               data = auto,
               kind = 'bar')

sns.factorplot(x = 'fuel-type',             # either x axis or y axis should be removed because count wont include both the axis
               hue = 'num-of-doors',
               col = 'engine-location',
               data = auto,
               kind = 'count')

sns.factorplot(x = 'fuel-type',
               y = 'horsepower',
               hue = 'num-of-doors',
               col = 'engine-location',
               data = auto,
               kind = 'box')

sns.factorplot(x = 'fuel-type',
               y = 'horsepower',
               hue = 'num-of-doors',
               col = 'engine-location',
               data = auto,
               kind = 'violin')

sns.factorplot(x = 'fuel-type',
               y = 'horsepower',
               hue = 'num-of-doors',
               col = 'engine-location',
               data = auto,
               kind = 'strip')

# regression plot
auto['peak-rpm'] = auto['peak-rpm'].replace('?', np.nan)
auto['peak-rpm'] = auto['peak-rpm'].astype('int')
auto['peak-rpm'] = auto['peak-rpm'].fillna(auto['peak-rpm'].mean())
sns.lmplot(x = 'horsepower', y = 'peak-rpm', data = auto)
sns.lmplot(x = 'horsepower', y = 'peak-rpm', data = auto, hue = 'fuel-type')

sns.regplot(x = 'playingtime', y = 'average_rating', data = games[games['playingtime'] < 500])