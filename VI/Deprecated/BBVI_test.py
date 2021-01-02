#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:05:58 2020

@author: dhulls
"""

import autograd.numpy as anp
from autograd import grad
from scipy.stats import norm
import numpy as np


## Autograd test

# def tanh(x):                 # Define a function
#     y = anp.exp(-2.0 * x)
#     return (1.0 - y) / (1.0 + y)

    
# grad_tanh = grad(tanh)
# A = grad_tanh(1.0)

## BBVI test for 1D Normal

Normal = norm(loc=3, scale=2)
data = Normal.rvs(100)

def func(x, param1, param2):
    y = anp.exp(-(x-param1)**2/(2*param2**2))
    return anp.log(1/(param2*anp.sqrt(2*anp.pi)) * y)


def func1(x, inp, param2):
    y = anp.exp(-(inp-x)**2/(2*param2**2))
    return anp.log(1/(param2*anp.sqrt(2*anp.pi)) * y)

def func2(x, inp, param1):
    y = anp.exp(-(inp-param1)**2/(2*x**2))
    return anp.log(1/(x*anp.sqrt(2*anp.pi)) * y)

t = 0
delta = 10000
tol = 0.0001
# req_vec = np.array([0,1])
batch = 100
param_sto = np.empty([1,2], dtype = float)
param_sto = np.concatenate((param_sto, np.array([0,1]).reshape(1,2)))
param_sto = np.delete(param_sto, 0, 0)
grad_F_mu = grad(func1)
grad_F_std = grad(func2)

t = 0
rho = 1.0
while delta > tol:
    t = t+1
    rnd1 = Normal.rvs(batch)
    grad_L_mu = 0.0
    grad_L_std = 0.0
    for ii in np.arange(0,batch,1):
        grad_L_mu = grad_L_mu + 1/batch * grad_F_mu(param_sto[t-1,0],rnd1[ii],param_sto[t-1,1]) * (np.log(Normal.pdf(rnd1[ii]))-func(rnd1[ii],param_sto[t-1,0],param_sto[t-1,1]))
        grad_L_std = grad_L_std + 1/batch * grad_F_std(param_sto[t-1,1],rnd1[ii],param_sto[t-1,0]) * (np.log(Normal.pdf(rnd1[ii]))-func(rnd1[ii],param_sto[t-1,0],param_sto[t-1,1]))
    
    param_sto = np.concatenate((param_sto, np.array([param_sto[t-1,0] + rho/t * grad_L_mu, param_sto[t-1,1] + rho/t * grad_L_std]).reshape(1,2)))
    delta = np.linalg.norm(param_sto[t,:]-param_sto[t-1,:]) / np.linalg.norm(param_sto[t-1,:])
    
        
    