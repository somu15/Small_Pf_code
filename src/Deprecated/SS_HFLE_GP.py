#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:27:21 2020

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

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from LimitStateFunctions import LimitStateFunctions as LSF
from ML_TF import ML_TF
from DrawRandom import DrawRandom as DR

Ndim = 1
LS1 = LSF()
DR1 = DR()
num_s = 500
value = 17 # (1)
# value = 1.8

## Subset simulation basic

# Nsims = 2500000
# y = np.zeros(Nsims)
# LS1 = LSF()
# DR1 = DR()

# for ii in np.arange(0,Nsims,1):
#     inp = (DR1.StandardNormal_Indep(N=Ndim))
#     inpp = inp[None,:]
#     y[ii] = LS1.Scalar_LS2_HF(inpp)
    
# req = len(np.rot90(np.where(y>value)))/Nsims

# req = 3.56e-5 (1)
# req = 6.92e-5 (2)

uni = uniform()
Nsub = 1500
Psub = 0.1
Nlim = 6
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,1,Nlim))
rv = norm(loc=0,scale=1)

for ii in np.arange(0,Nsub,1):
    inp = (DR1.StandardNormal_Indep(N=Ndim))
    inpp = inp[None,:]
    y1[ii,0] = LS1.Scalar_LS2_HF(inpp)
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
            rv1 = norm(loc=inp1[ii-(int(Psub*Nsub)),jj,kk],scale=1.0)
            prop = (rv1.rvs())
            r = rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
            if r>uni.rvs():
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ii-(int(Psub*Nsub)),jj,kk]
            inpp[jj] = nxt[0,jj]
        # inpp = inpp[None,:]
        # inpp = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])[None,:]
        y_nxt = LS1.Scalar_LS2_HF(inpp[None,:])
        if y_nxt>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = inp1[ii-(int(Psub*Nsub)),:,kk]
            y1[ii,kk] = y1[ii-(int(Psub*Nsub)),kk]

Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/Nsub
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)

## Subset simultion with HF-LF and GP

Ninit_GP = 50
y_LF_GP = np.empty(1, dtype = float)
y_HF_GP = np.empty(1, dtype = float)
inp_GPtrain = np.empty(1, dtype = float)
for ii in np.arange(0,Ninit_GP,1):
    inp = (DR1.StandardNormal_Indep(N=Ndim))
    inpp = inp[None, :]
    inp_GPtrain = np.concatenate((inp_GPtrain, inp))
    y_LF_GP = np.concatenate((y_LF_GP, LS1.Scalar_LS2_LF(inpp)))
    y_HF_GP = np.concatenate((y_HF_GP, LS1.Scalar_LS2_HF(inpp)))

ML = ML_TF(obs_ind = inp_GPtrain[:,None], obs = (y_HF_GP-y_LF_GP))
amp1, len1, var1 = ML.GP_train()
samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inp_GPtrain[:,None], num_samples=num_s)

uni = uniform()
Nsub = 1200
Psub = 0.1
Nlim = 5
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,1,Nlim))
rv = norm(loc=0,scale=1)
# u_lim_vec = np.array([2,3,4,5])
u_lim_vec = np.array([2,2,2,2,2,2])
# u_lim_vec = np.array([2,2,3.5,4.5])
# u_lim = 3.5

u_GP = np.empty(1, dtype = float)
var_GP = np.empty(1, dtype = float)
var_GP[0] = var1.numpy().reshape(1)
subs_info = np.empty(1, dtype = float)
subs_info[0] = np.array(0).reshape(1)
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)

for ii in np.arange(0,Nsub,1):
    inp = DR1.StandardNormal_Indep(N=Ndim)
    inpp = inp[None,:]
    LF = LS1.Scalar_LS2_LF(inpp)
    inp1[ii,:,0] = inp
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inpp, num_samples=num_s)
    GP_diff = np.mean(np.array(samples1),axis=0)
    u_check = np.abs(LF + GP_diff)/np.std(np.array(samples1),axis=0)
    u_GP = np.concatenate((u_GP, u_check))
    u_lim = u_lim_vec[0]
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = LS1.Scalar_LS2_HF(inpp)
        inp_GPtrain = np.concatenate((inp_GPtrain, inp))
        y_LF_GP = np.concatenate((y_LF_GP, LF))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
        GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
        # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
        ML = ML_TF(obs_ind = inp_GPtrain[:,None], obs = (y_HF_GP-y_LF_GP))
        amp1, len1, var1 = ML.GP_train()
        var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
        subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))

# inpp = np.zeros(Ndim)
for kk in np.arange(1,Nlim,1):
    y1[0:(int(Psub*Nsub)-1),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1)-1)]
    y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)-1),kk])
    indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub)-1)]
    inp1[0:(int(Psub*Nsub)-1),:,kk] = inp1[indices,:,kk-1]
    for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
        nxt = np.zeros((1,Ndim))
        for jj in np.arange(0,Ndim,1):
            rv1 = norm(loc=inp1[ii-(int(Psub*Nsub)),jj,kk],scale=1.0)
            prop = (rv1.rvs())
            r = rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
            if r>uni.rvs():
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ii-(int(Psub*Nsub)),jj,kk]
            inpp[jj] = nxt[0,jj]
        # inpp = inpp[None,:]
        # inpp = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])[None,:]
        LF = LS1.Scalar_LS2_LF(inpp[None,:])
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inpp, num_samples=num_s)
        GP_diff = np.mean(np.array(samples1),axis=0)
        u_check = np.abs(LF + GP_diff)/np.std(np.array(samples1),axis=0)
        u_GP = np.concatenate((u_GP, u_check[:,0]))
        u_lim = u_lim_vec[kk]
        if u_check > u_lim:
            y_nxt = LF + GP_diff
        else:
            y_nxt = LS1.Scalar_LS2_HF(inpp)
            inp_GPtrain = np.concatenate((inp_GPtrain, inp))
            y_LF_GP = np.concatenate((y_LF_GP, LF[:,0]))
            y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF[:,0] + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            ML = ML_TF(obs_ind = inp_GPtrain[:,None], obs = (y_HF_GP-y_LF_GP))
            amp1, len1, var1 = ML.GP_train()
            var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
            subs_info = np.concatenate((subs_info, np.array(kk).reshape(1)))
        if y_nxt>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = inp1[ii-(int(Psub*Nsub)),:,kk]
            y1[ii,kk] = y1[ii-(int(Psub*Nsub)),kk]
            
Pf = 1
Pi_sto = np.zeros(Nlim)
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/Nsub
    Pi_sto[kk] = Pi
    Pf = Pf * Pi
