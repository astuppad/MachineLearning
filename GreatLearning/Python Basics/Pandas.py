# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:09:35 2019

@author: N827941
"""

import numpy as np
import pandas as pd

labels = ['w', 'x', 'y', 'z']
list = [10,20,30,40]
array = np.array([10,20,30,40])
dict =  {'w':10, 'x':20, 'y':30, 'z':40}

# creating the series for list
pd.Series(data = list)
pd.Series(data = list, index = labels)

#creating the series for the array
pd.Series(data = array)
pd.Series(data = array, index = labels)

# Series indexing
sports1 = pd.Series([1,2,3,4], index = ['cricket', 'football', 'basketball', 'volleyball'])
sports1['cricket']
sports1[0]

sports2 = pd.Series([1,2,5,6], index = ['cricket', 'football', 'baseball', 'basketball'])
sports = sports1 + sports2    #Collection of Series will share the common index
sports

from numpy.random import randn
np.random.seed(1) # To generate same random value every time.
dataframe = pd.DataFrame(randn(10, 5), index = 'A B C D E F G H I J'.split(), columns = 'Score1 Score2 Score3 Score4 Score5'.split())
dataframe
dataframe['Score4']
dataframe[['Score2', 'Score1']]
dataframe['Score6'] = dataframe['Score1'] + dataframe['Score2']
dataframe
dataframe.drop('Score6', axis=1) #Drop the column or row from the dataframe axis = 1 for column axis = 0 for row by default axis = 0
dataframe.drop('Score6', axis=1, inplace = True) # if the inplace is true it does not create the view

dataframe.loc['A'] # prints all the score values for the 'A' row
dataframe.iloc[1] # normal index values
dataframe.loc['A', 'Score2'] # gives the value present in a specific row and specific column
dataframe.loc[['A', 'B'],['Score1','Score2']] # subsetting
dataframe[dataframe > 0.5]
dataframe[dataframe['Score1'] > 0.5]
dataframe[dataframe['Score1'] > 0.5]['Score2']
dataframe[(dataframe['Score1'] > 0.5) & (dataframe['Score2'] > 0.0)]
dataframe[dataframe['Score1'] > 0.5][0:2]

# reset the indexes
dataframe.reset_index() # Converts user given indexes to new column and keeps default indexes 
newindex = 'IND JP CAN GE IT PL FY IU RT IP'.split()
dataframe['Countries'] = newindex
dataframe

dataframe.set_index('Countries', inplace = True)
dataframe

# Missing Values
dataframe = pd.DataFrame({'cricket': [1, np.nan, 3, np.nan, 5, 6],
                          'football': [1, np.nan, 3, np.nan, 5, 6],
                          'basketball': [1, 2, 3, 4, 5, 6]}) 
dataframe.dropna() # drops the null rows from the dataframe
dataframe.dropna(axis = 1) # drops the column from the datafram
dataframe.dropna(thresh = 2) # drops if the rows are not rows are having more than 2 values not nan 

dataframe.fillna(value = 0) # filling the missing values
dataframe['cricket'] = dataframe.fillna(value = dataframe['cricket'].mean()) # filling the missing values with mean

dat = {'CustID' : ['1001', '1001', '1002', '1002', '1003', '1003'],
       'CustName' : ['UIPat', 'DatRob', 'Goog', 'Chrysler', 'Ford', 'GM'],
       'ProfitinLakhs' : [1005, 3245, 1245, 8765, 5463, 3547]
       }
dat
dataframe = pd.DataFrame(dat)

# Group by 
grouped_CustID = dataframe.groupby('CustID')
grouped_CustID.mean()
grouped_CustID.std()
grouped_CustID.describe()
grouped_CustID.describe().transpose()
grouped_CustID.describe().transpose()['1002']

# Merging Similar to SQL logic
dafa1 = pd.DataFrame({'CustID' : ['101', '102', '103', '104'],
                      'Sales' : [123456, 45321, 54385, 53212],
                      'Priority': ['CAT0', 'CAT1', 'CAT2', 'CAT3'],
                      'Prime': ['yes', 'no', 'no', 'yes']},
                      index= [0, 1, 2, 3])
dafa2 = pd.DataFrame({'CustID' : ['101', '103', '104', '105'],
                      'Sales' : [123456, 54385, 53212, 4534],
                      'Priority': ['CAT4', 'CAT5', 'CAT6', 'CAT7'],
                      'Prime': ['yes', 'no', 'no', 'no']},
                      index= [4, 5, 6, 7])
pd.concat([dafa1, dafa2]) # Concatinating by column
pd.concat([dafa1, dafa2], axis = 1) # Concatinating by row

# Join vs Merge
# Merge is function which joins by column that you specify
dafa1.merge(dafa2, how = 'outer', on = 'CustID')


dafa3 = pd.DataFrame({ 'Q1' : ['101', '102', '103'],
                      'Q2' : ['201', '202', '203']},
                        index = ['I1', 'I2', 'I3'])
dafa4 = pd.DataFrame({ 'Q3' : ['301', '302', '303'],
                      'Q4' : ['401', '402', '403']},
                        index = ['I1', 'I2', 'I4'])
# Join is function which joins by index
dafa3.join(dafa4, how = 'outer')
dafa3.join(dafa4, how = 'inner')
dafa3.join(dafa4, how = 'left')
dafa3.join(dafa4, how = 'right')

    
