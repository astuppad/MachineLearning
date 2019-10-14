# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 08:43:09 2019

@author: N827941
"""

# Lambda function to multipy number by 2
double = lambda x: x * 2
double(9)

# Program to filter out only even number from the list
my_list = [11, 12, 4, 13, 16, 14, 18, 1]
new_list = list(filter(lambda x: (x%2 == 0), my_list))
print(new_list)

# use to identify the null values
import pandas as pd
import numpy as np

mpg_df = pd.read_csv("car-mpg.csv")
mpg_df = mpg_df.replace('?', np.nan)
mpg_df['hp'] = mpg_df['hp'].astype('float64')
numeric_cols = mpg_df.drop('car_name', axis = 1)
print(numeric_cols.head(50)) 
numeric_cols = numeric_cols.apply(lambda x: x.fillna(x.median()), axis = 0) 
print(numeric_cols.head(50))
mpg_df.hp.median()
 