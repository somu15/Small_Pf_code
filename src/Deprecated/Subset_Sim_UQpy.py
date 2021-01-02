#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 00:11:44 2020

@author: dhulls
"""

### Basic subset simulation

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
from mlxtend.plotting import scatterplotmatrix
from LimitStateFunctions import LimitStateFunctions as LSF
from DrawRandom import DrawRandom as DR


## Initial setup

Nsims = 5000000
y = np.zeros(Nsims)
# value = 4.5 # LS2
value = 1200 # LS4
Ndim = 3
LS1 = LSF()
DR1 = DR()

# for ii in np.arange(0,Nsims,1):
#     inp = DR1.StandardNormal_Indep(N=Ndim)
#     inpp = inp[None,:]
#     y[ii] = LS1.Scalar_LS4(inpp)
    
# req = len(np.rot90(np.where(y>value)))/Nsims

# req = 0.000131 # For LS2

req = 5.72e-5 # For LS4

## Subset simulation

uni = uniform()
Nsub = 5000
Psub = 0.1
Nlim = 4
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,3,Nlim))
rv = norm(loc=0,scale=1)

for ii in np.arange(0,Nsub,1):
    inp = DR1.StandardNormal_Indep(N=Ndim)
    inpp = inp[None,:]
    y1[ii,0] = LS1.Scalar_LS4(inpp)
    inp1[ii,:,0] = inp

inpp = np.zeros(Ndim)
for kk in np.arange(1,Nlim,1):
    y1[0:(int(Psub*Nsub)-1),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1)-1)]
    y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)-1),kk])
    indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub)-1)]
    inp1[0:(int(Psub*Nsub)-1),:,kk] = inp1[indices,:,kk-1]
    for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
        nxt = np.zeros((1,Ndim))
        for jj in np.arange(0,Ndim,1):
            rv1 = norm(loc=inp1[ii-(int(Psub*Nsub)),jj,kk],scale=1)
            prop = rv1.rvs()
            r = rv.pdf(prop)/rv.pdf(inp1[ii-(int(Psub*Nsub)),jj,kk])
            if r>uni.rvs():
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ii-(int(Psub*Nsub)),jj,kk]
            inpp[jj] = nxt[0,jj]
        # inpp = inpp[None,:]
        # inpp = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])[None,:]
        y_nxt = LS1.Scalar_LS4(inpp[None,:])
        if y_nxt>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
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

## Plotting data

all_inps = np.zeros((Nsub*Nlim,Ndim+1))
indc = np.zeros(Nsub*Nlim)

for ii in np.arange(0,Nlim,1):
    all_inps[(Nsub*ii):((ii+1)*Nsub),0:Ndim] = inp1[:,:,ii]
    all_inps[(Nsub*ii):((ii+1)*Nsub),Ndim] = y1[:,ii]
    indc[(Nsub*ii):((ii+1)*Nsub)] = ii*np.ones(len(y1[:,ii]))

names = ['X1', 'X2',
          'X3', 'Y']

fig, axes = scatterplotmatrix(all_inps[indc==0], figsize=(10, 8), alpha=0.5)
fig, axes = scatterplotmatrix(all_inps[indc==1], fig_axes=(fig, axes), alpha=0.5)
fig, axes = scatterplotmatrix(all_inps[indc==2], fig_axes=(fig, axes), alpha=0.5)
fig, axes = scatterplotmatrix(all_inps[indc==3], fig_axes=(fig, axes), alpha=0.5, names=names)

plt.tight_layout()
plt.show()
