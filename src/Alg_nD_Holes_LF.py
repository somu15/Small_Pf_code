#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:12:04 2021

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
from scipy.stats import cauchy
import matplotlib.pyplot as plt
from UQpy.SampleMethods import MH
from UQpy.Distributions import Distribution
import time
from UQpy.Distributions import Normal
from UQpy.SampleMethods import MMH

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from LimitStateFunctions import LimitStateFunctions as LSF
from ML_TF import ML_TF
from DrawRandom import DrawRandom as DR
from pyDOE import *

Ndim = 3
value = 100.0 # 600.0

LS1 = LSF()
DR1 = DR()
num_s = 500

## Monte Carlo simulations

# Nsims = int(5000)
# inps = np.zeros((Nsims,Ndim))
# yns = np.zeros(Nsims)
# ys = np.zeros(Nsims)
# LS1 = LSF()
# DR1 = DR()

# for ii in np.arange(0,Nsims,1):
#     inp = (DR1.FluidRandom())
#     inpp = inp[None,:]
#     inps[ii,:] = inp
#     yns[ii] = np.array(LS1.Fluid_NS(inpp))
#     ys[ii] = np.array(LS1.Fluid_S(inpp))
#     print(ii/Nsims)

# req = len(np.rot90(np.where(y>value)))/Nsims

# req = 0.0001469 (2.4e6 simulations)

# Basic subset simulation

LS1 = LSF()
DR1 = DR()
num_s = 500

uni = uniform()
Nsub = 100
Psub = 0.1
Nlim = 5
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
# y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
y_seed = np.zeros(int(Psub*Nsub))

for ii in np.arange(0,Nsub,1):
    inp = (DR1.HolesRandom())
    inpp = inp[None,:]
    y1[ii,0] = np.array(LS1.Holes_HF(inpp))
    inp1[ii,:,0] = inp
    print(ii)

inpp = np.zeros(Ndim)
count_max = Nsub/(Psub*Nsub)
count = 100000
ind_max = 1
r_sto = np.zeros((Nsub-int(Psub*Nsub),Nlim-1,Ndim))
# ind_sto = 3
prop_std_req =np.array([0.375,0.375,0.375])

for kk in np.arange(1,Nlim,1):
    ind_max = 0
    ind_sto = -1
    count = np.inf
    y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
    indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
    inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
    for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
        print(kk)
        print(ii)
        nxt = np.zeros((1,Ndim))
        if count > count_max:
            # ind_max = random.randint(0,int(Psub*Nsub)) # ind_sto
            ind_sto = ind_sto + 1
            ind_max = ind_sto
            count = 0
        else:
            ind_max = ii-1
            
        count = count + 1

        for jj in np.arange(0,Ndim,1):
            # rv1 = norm(loc=np.log(inp1[ind_max,jj,kk]),scale=0.5)
            # prop = np.exp(rv1.rvs())
            rv1 = uniform(loc=(np.log(inp1[ind_max,jj,kk])-prop_std_req[jj]),scale=(2*prop_std_req[jj]))
            prop = np.exp(rv1.rvs())
            r = np.log(DR1.HolesPDF(rv_req=(prop), index=jj)) - np.log(DR1.HolesPDF(rv_req=(inp1[ind_max,jj,kk]),index=jj)) # rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
            r_sto[ii-(int(Psub*Nsub)),kk-1,jj] = r
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ind_max,jj,kk]
            inpp[jj] = nxt[0,jj]
        y_nxt = np.array(LS1.Holes_HF(inpp[None,:])).reshape(1)
        if y_nxt>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = inp1[ind_max,:,kk]
            y1[ii,kk] = y1[ind_max,kk]

Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)

