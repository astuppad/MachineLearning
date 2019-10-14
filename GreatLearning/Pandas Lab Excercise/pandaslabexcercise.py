# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:41:56 2019

@author: N827941
"""

import pandas as pd
auto = pd.read_csv('Automobile.csv')

auto.head()

# to know rows and columns
auto.info()

auto['price'].mean()
(auto[auto['price']==auto['price'].min()])['make']
(auto[auto['price']==auto['price'].max()])['make']

(auto[auto['horsepower'] > 100])['horsepower'].count()
auto[auto['body_style'] == 'hatchback'].info()
auto['make'].value_counts().head(3)
auto[auto['price'] == 7099]['make']
auto[auto['price'] > 40000]['make'].unique()
auto[(auto['body_style'] == 'sedan') & (auto['price'] < 7000)]
