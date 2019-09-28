# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:51:01 2019

@author: N827941
"""

import numpy as np
import pandas as pd
from sklearn import tree

input_file = './resume_shortlist.csv'
df = pd.read_csv(input_file, header = 0)

d = {'Y' : 1, 'N' : 0}
df['Shortlisted'] = df['Shortlisted'].map(d)
df['Referral'] = df['Referral'].map(d)

d = {'Mtech' : 1, 'BTech' : 0}
df['Degree'] = df['Degree'].map(d)

features =  list(df.columns[:4])

y = df['Shortlisted']
x = df[features]
classifier = tree.DecisionTreeClassifier()
classifier.fit(x, y)
