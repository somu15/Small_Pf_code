#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 21:34:21 2020

@author: som
"""

from os import sys
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import random
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import uniform
import matplotlib.pyplot as plt
from UQpy.SampleMethods import MH
from UQpy.Distributions import Distribution
import time

## Limit state function

def LS(x):
    return np.sqrt(np.sum(np.power(x[0],2)+np.power(x[1],2)+np.power(x[2],2)))

## Test limit state function

Nsims = 1000000
y = np.zeros(Nsims)
value = 4.5

rv = norm(loc=0,scale=1)
# for ii in np.arange(0,Nsims,1):
#     inp = np.array([rv.rvs(), rv.rvs(), rv.rvs()])
#     y[ii] = LS(inp)
    
# req = len(np.rot90(np.where(y>4.5)))/Nsims

req = 0.000131

## Subset simulation

uni = uniform()
Nsub = 1000
y1 = np.zeros(Nsub)
inp1 = np.zeros((Nsub,3))
y2 = np.zeros(Nsub)
inp2 = np.zeros((Nsub,3))
y3 = np.zeros(Nsub)
inp3 = np.zeros((Nsub,3))
y4 = np.zeros(Nsub)
inp4 = np.zeros((Nsub,3))
y5 = np.zeros(Nsub)
inp5 = np.zeros((Nsub,3))
for ii in np.arange(0,Nsub,1):
    inp = np.array([rv.rvs(), rv.rvs(), rv.rvs()])
    y1[ii] = LS(inp)
    inp1[ii,:] = inp

y2[0:99] = np.sort(y1)[900:999]
y2_lim = np.min(y2[0:99])
indices = (-y1).argsort()[:99]
inp2[0:99,:] = inp1[indices,:]
for ii in np.arange(100,(Nsub),1):
    nxt = np.zeros((1,3))
    for jj in np.arange(0,3,1):
        rv1 = norm(loc=inp2[ii-100,jj],scale=1)
        prop = rv1.rvs()
        r = rv.pdf(prop)/rv.pdf(inp2[ii-100,jj])
        if r>uni.rvs():
            nxt[0,jj] = prop
        else: 
            nxt[0,jj] = inp2[ii-100,jj]
    y_nxt = LS(np.array([nxt[0,0], nxt[0,1], nxt[0,2]]))
    if y_nxt>y2_lim:
        inp2[ii,:] = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
        y2[ii] = y_nxt
    else:
        inp2[ii,:] = inp2[ii-100,:]
        y2[ii] = y2[ii-100]


y3[0:99] = np.sort(y2)[900:999]
y3_lim = np.min(y3[0:99])
indices = (-y2).argsort()[:99]
inp3[0:99,:] = inp2[indices,:]
for ii in np.arange(100,(Nsub),1):
    nxt = np.zeros((1,3))
    for jj in np.arange(0,3,1):
        rv1 = norm(loc=inp3[ii-100,jj],scale=1)
        prop = rv1.rvs()
        r = rv.pdf(prop)/rv.pdf(inp3[ii-100,jj])
        if r>uni.rvs():
            nxt[0,jj] = prop
        else: 
            nxt[0,jj] = inp3[ii-100,jj]
    y_nxt = LS(np.array([nxt[0,0], nxt[0,1], nxt[0,2]]))
    if y_nxt>y3_lim:
        inp3[ii,:] = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
        y3[ii] = y_nxt
    else:
        inp3[ii,:] = inp3[ii-100,:]
        y3[ii] = y3[ii-100]

y4[0:99] = np.sort(y3)[900:999]
y4_lim = np.min(y4[0:99])
indices = (-y3).argsort()[:99]
inp4[0:99,:] = inp3[indices,:]
for ii in np.arange(100,(Nsub),1):
    nxt = np.zeros((1,3))
    for jj in np.arange(0,3,1):
        rv1 = norm(loc=inp4[ii-100,jj],scale=1)
        prop = rv1.rvs()
        r = rv.pdf(prop)/rv.pdf(inp4[ii-100,jj])
        if r>uni.rvs():
            nxt[0,jj] = prop
        else: 
            nxt[0,jj] = inp4[ii-100,jj]
    y_nxt = LS(np.array([nxt[0,0], nxt[0,1], nxt[0,2]]))
    if y_nxt>y4_lim:
        inp4[ii,:] = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
        y4[ii] = y_nxt
    else:
        inp4[ii,:] = inp4[ii-100,:]
        y4[ii] = y4[ii-100]
        

Pf = len(np.rot90(np.where(y1>y2_lim))) * len(np.rot90(np.where(y2>y3_lim))) * len(np.rot90(np.where(y3>y4_lim))) * len(np.rot90(np.where(y4>value)))
Pf = Pf/(1000*1000*1000*1000)


# def rosenbrock_no_params(x):
#      return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/20)
 

# x = MH(dimension=2, pdf_target=rosenbrock_no_params, nburn=500, jump=50, nsamples=500, nchains=1)
# print(x.samples.shape)
# plt.plot(x.samples[:,0], x.samples[:,1], 'o', alpha=0.5)
# plt.show()