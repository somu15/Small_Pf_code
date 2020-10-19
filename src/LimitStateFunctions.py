#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:55:13 2020

@author: dhulls
"""

import numpy as np

class LimitStateFunctions:
    
    # def __init__(self):
    #     # self.Input_vec = Input_vec
    
    def Scalar_LS1(self, Input_vec=None): # sin(sqrt(x1^2 + x2^2))
        return np.sin(np.sqrt(Input_vec[:,0]**2+Input_vec[:,1]**2))
    
    def Scalar_LS2(self, Input_vec=None): # sqrt(x1^2 + x2^2 + x3^2)
        return np.sqrt((Input_vec[:,0]**2+Input_vec[:,1]**2+Input_vec[:,2]**2))
    
    def Scalar_LS3(self, Input_vec=None): # sin(3 * pi * x)
        return np.sin(3 * np.pi * Input_vec[:,0])
    
    def Scalar_LS4(self, Input_vec=None): # sum(xi^5) - 5 * sum(xi) + 12
        return (Input_vec[:,0]**5+Input_vec[:,1]**5+Input_vec[:,2]**5) - 5 * (Input_vec[:,0]+Input_vec[:,1]+Input_vec[:,2]) + 12
