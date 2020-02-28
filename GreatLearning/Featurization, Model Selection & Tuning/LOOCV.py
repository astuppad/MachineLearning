# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:05:25 2020

@author: Niket
"""

from numpy import array

from sklearn.model_selection import LeaveOneOut

data = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

loocv = LeaveOneOut()

for train, test in loocv.split(data):
    print('train:%s, test %s' % (data[train], data[test]))