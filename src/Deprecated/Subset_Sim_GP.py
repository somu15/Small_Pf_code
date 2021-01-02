#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:53:29 2020

@author: dhulls
"""

### Subset simulation using Gaussian Process surrogate

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

## INIT

# req = 0.000131 # LS2
req = 5.72e-5 # For LS4
# value = 4.5 # LS2
value = 1200 # LS4
rv = norm(loc=0,scale=1)
Ndim = 3
LS1 = LSF()
DR1 = DR()

## SS with GP

uni = uniform()
Nsub = 5000
Ntrain = 500
Psub = 0.1
Nlim = 4
num_s = 50
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,3,Nlim))

amp1 = 1 
len1 = 1 
var1 = 1

tmp = np.zeros((Nsub-Ntrain))
tmp1 = np.zeros((Nsub-Ntrain))
for ii in np.arange(0,Nsub,1):
    if ii<Ntrain:
        inp = DR1.StandardNormal_Indep(N=Ndim)
        inpp = inp[None,:]
        y1[ii,0] = LS1.Scalar_LS4(inpp)
        inp1[ii,:,0] = inp
    elif ii==Ntrain:
        inp = DR1.StandardNormal_Indep(N=Ndim)
        inpp = inp[None,:]
        y1[ii,0] = LS1.Scalar_LS4(inpp)
        inp1[ii,:,0] = inp
        ML = ML_TF(obs_ind = inp1[0:Ntrain,:,0], obs = y1[0:Ntrain,0])
        amp1, len1, var1 = ML.GP_train()
    else:
        inp = DR1.StandardNormal_Indep(N=Ndim)
        inpp = inp[None,:]
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inpp, num_samples=num_s)
        y1[ii,0] = np.mean(np.array(samples1))
        tmp[ii-Ntrain] = np.mean(np.array(samples1))
        tmp1[ii-Ntrain] = LS1.Scalar_LS4(inpp)
        inp1[ii,:,0] = inp

inpp = np.zeros(Ndim)
save_dat_GP = np.zeros(((Nsub-int(Psub*Nsub)-Ntrain+1), Nlim))
save_dat_EX = np.zeros(((Nsub-int(Psub*Nsub)-Ntrain+1), Nlim))
save_dat_GP[:,0] = tmp[0:(Nsub-int(Psub*Nsub)-Ntrain+1)]
save_dat_EX[:,0] = tmp1[0:(Nsub-int(Psub*Nsub)-Ntrain+1)]
for kk in np.arange(1,Nlim,1):
    y1[0:(int(Psub*Nsub)-1),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1)-1)]
    y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)-1),kk])
    indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub)-1)]
    inp1[0:(int(Psub*Nsub)-1),:,kk] = inp1[indices,:,kk-1]
    inp_gp = np.zeros((Ntrain+1, 3))
    out_gp = np.zeros(Ntrain+1)
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
        # inpp = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])[None,:]
        y_nxt = 0
        if ii<(int(Psub*Nsub)+Ntrain):
            inp_gp[ii-int(Psub*Nsub),:] = inpp[None,:]
            y_nxt = LS1.Scalar_LS4(inpp[None,:])
            out_gp[ii-int(Psub*Nsub)] = y_nxt
        elif ii==(int(Psub*Nsub)+Ntrain):
            inp_gp[ii-int(Psub*Nsub),:] = inpp[None,:]
            y_nxt = LS1.Scalar_LS4(inpp[None,:])
            out_gp[ii-int(Psub*Nsub)] = y_nxt
            ML = ML_TF(obs_ind = inp_gp, obs = out_gp)
            amp1, len1, var1 = ML.GP_train()
        else:
            samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inpp[None,:], num_samples=num_s)
            y_nxt = np.mean(np.array(samples1))
            save_dat_GP[(ii-int(Psub*Nsub)-Ntrain),kk] = y_nxt
            save_dat_EX[(ii-int(Psub*Nsub)-Ntrain),kk] = LS1.Scalar_LS4(inpp[None,:])
        # y_nxt = LS1.Scalar_LS4(inpp)
        if y_nxt>y1_lim[kk-1]:
            inp1[ii,:,kk] = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = inp1[ii-(int(Psub*Nsub)),:,kk]
            y1[ii,kk] = y1[ii-(int(Psub*Nsub)),kk]
            
Pf = 1
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/Nsub
    Pf = Pf * Pi
    
## Plotting

plt.scatter(save_dat_EX[0:4000,0], save_dat_GP[0:4000,0], alpha=0.2,
            s=25, c='red', label='Subset 1')
plt.xlim(0, 1200)
plt.ylim(0, 1200)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('True Y')
plt.ylabel('GP predicted Y')
plt.title('Subset 1')
# plt.legend(loc='lower left')
plt.show()

plt.scatter(save_dat_EX[0:4000,1], save_dat_GP[0:4000,1], alpha=0.2,
            s=25, c='blue', label='Subset 1')
plt.xlim(0, 1200)
plt.ylim(0, 1200)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('True Y')
plt.ylabel('GP predicted Y')
plt.title('Subset 2')
# plt.legend(loc='lower left')
plt.show()

plt.scatter(save_dat_EX[0:4000,2], save_dat_GP[0:4000,2], alpha=0.2,
            s=25, c='green', label='Subset 1')
plt.xlim(0, 1200)
plt.ylim(0, 1200)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('True Y')
plt.ylabel('GP predicted Y')
plt.title('Subset 3')
# plt.legend(loc='lower left')
plt.show()

plt.scatter(save_dat_EX[0:4000,3], save_dat_GP[0:4000,3], alpha=0.2,
            s=25, c='black', label='Subset 1')
plt.xlim(0, 1200)
plt.ylim(0, 1200)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('True Y')
plt.ylabel('GP predicted Y')
plt.title('Subset 4')
# plt.legend(loc='lower left')
plt.show()
