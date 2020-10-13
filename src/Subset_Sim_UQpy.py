#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 00:11:44 2020

@author: dhulls
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
from UQpy.Distributions import Normal
from UQpy.SampleMethods import MMH

def rosenbrock_no_params(x):
      return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/20)
 

# x = MH(dimension=2, pdf_target=rosenbrock_no_params, nburn=500, jump=50, nsamples=500, nchains=1)
# print(x.samples.shape)
# plt.plot(x.samples[:,0], x.samples[:,1], 'o', alpha=0.5)
# plt.show()

# proposal = [Normal(), Normal()]
# proposal_is_symmetric = [False, False]

# x = MMH(dimension=2, nburn=1, jump=50, nsamples=1, pdf_target=rosenbrock_no_params,
#         proposal=proposal, proposal_is_symmetric=proposal_is_symmetric, nchains=1)

# fig, ax = plt.subplots()
# ax.plot(x.samples[:, 0], x.samples[:, 1], linestyle='none', marker='.')

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
Nsub = 5000
Psub = 0.1
Nlim = 4
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,3,Nlim))

for ii in np.arange(0,Nsub,1):
    inp = np.array([rv.rvs(), rv.rvs(), rv.rvs()])
    y1[ii,0] = LS(inp)
    inp1[ii,:,0] = inp


for kk in np.arange(1,Nlim,1):
    y1[0:(int(Psub*Nsub)-1),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1)-1)]
    y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)-1),kk])
    indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub)-1)]
    inp1[0:(int(Psub*Nsub)-1),:,kk] = inp1[indices,:,kk-1]
    for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
        nxt = np.zeros((1,3))
        for jj in np.arange(0,3,1):
            rv1 = norm(loc=inp1[ii-(int(Psub*Nsub)),jj,kk],scale=1)
            prop = rv1.rvs()
            r = rv.pdf(prop)/rv.pdf(inp1[ii-(int(Psub*Nsub)),jj,kk])
            if r>uni.rvs():
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ii-(int(Psub*Nsub)),jj,kk]
        y_nxt = LS(np.array([nxt[0,0], nxt[0,1], nxt[0,2]]))
        if y_nxt>y1_lim[kk-1]:
            inp1[ii,:,kk] = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = inp1[ii-(int(Psub*Nsub)),:,kk]
            y1[ii,kk] = y1[ii-(int(Psub*Nsub)),kk]
            


Pf = 1
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/Nsub
    Pf = Pf * Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)

