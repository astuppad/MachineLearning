# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:00:17 2019

@author: N827941
"""
file_path = './height-weight.csv'

import pandas as pd
from pandas import DataFrame as df

data_frame = pd.read_csv(file_path, sep = ',')

data_frame['height'] = data_frame['Height(Inches)'].map(lambda inches: round(inches*0.833333, 2))

data_frame['weight'] = data_frame['Weight(Pounds)'].map(lambda pounds: round(pounds*0.453592, 2))

df = data_frame[['height', 'weight']]

import matplotlib.pyplot as plt

plt.scatter(df['height'], df['weight'])
plt.title('Height Weight Scatter Plot')
plt.xlabel('Height (in feet)')
plt.ylabel('Weight (in pounds)')
plt.show()

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(df['height'], df['weight'])

def predict(x):
        return slope * x + intercept

fitLine = predict(df['height'])

plt.scatter(df['height'], df['weight'])
plt.plot(df['height'], fitLine, c = 'r')
plt.show()
    
predict(4)

r_value ** 2