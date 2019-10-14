# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 09:10:21 2019

@author: N827941
"""

 
class Class1:
    name = "Niket"
    def function(self):
        print("The name " + self.name + " is inside the class.")
        
c1 = Class1()
c1.function()

c2 = Class1()
c2.name = "Alok"
c2.function()
