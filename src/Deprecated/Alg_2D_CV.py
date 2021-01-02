#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:56:54 2020

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

Ndim = 2
value = 0.0

def Convert(lst): 
    return [ -i for i in lst ] 

## Basic subset simulation

# LS1 = LSF()
# DR1 = DR()
# num_s = 500

# uni = uniform()
# Nsub = 5000
# Psub = 0.1
# Nlim = 2
# y1 = np.zeros((Nsub,Nlim))
# y1_lim = np.zeros(Nlim)
# y1_lim[Nlim-1] = value
# inp1 = np.zeros((Nsub,Ndim,Nlim))
# rv = norm(loc=0,scale=1)

# for ii in np.arange(0,Nsub,1):
#     inp = (DR1.StandardNormal_Indep(N=Ndim))
#     inpp = inp[None,:]
#     y1[ii,0] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp)))
#     inp1[ii,:,0] = inp

# inpp = np.zeros(Ndim)
# count_max = int(Nsub/(Psub*Nsub))

# for kk in np.arange(1,Nlim,1):
#     ind_max = 0
#     ind_sto = -1
#     count = np.inf
#     y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
#     y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
#     indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
#     inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
#     for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
#         nxt = np.zeros((1,Ndim))
#         if count > count_max:
#             # ind_max = random.randint(0,int(Psub*Nsub)) # ind_sto
#             ind_sto = ind_sto + 1
#             ind_max = ind_sto
#             count = 0
#         else:
#             ind_max = ii-1
            
#         count = count + 1
                
#         for jj in np.arange(0,Ndim,1):
#             rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
#             prop = (rv1.rvs())
#             r = rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
#             if r>uni.rvs():
#                 nxt[0,jj] = prop
#             else: 
#                 nxt[0,jj] = inp1[ii-(int(Psub*Nsub)),jj,kk]
#             inpp[jj] = nxt[0,jj]
#         # inpp = inpp[None,:]
#         # inpp = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])[None,:]
#         y_nxt = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp[None,:]))).reshape(1)
#         if y_nxt>y1_lim[kk-1]:
#             inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
#             y1[ii,kk] = y_nxt
#         else:
#             inp1[ii,:,kk] = inp1[ii-(int(Psub*Nsub)),:,kk]
#             y1[ii,kk] = y1[ii-(int(Psub*Nsub)),kk]

# Pf = 1
# Pi_sto = np.zeros(Nlim)
# cov_sq = 0
# for kk in np.arange(0,Nlim,1):
#     Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
#     Pf = Pf * Pi
#     Pi_sto[kk] = Pi
#     cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
# cov_req = np.sqrt(cov_sq)


## SS with HF and LFGP, and GP diff

LS1 = LSF()
DR1 = DR()
num_s = 500

## Training GP

def Norm1(X1,X):
    return X1 # (X1-np.mean(X,axis=0))/(np.std(X,axis=0))

def InvNorm1(X1,X):
    return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

Ninit_GP = 50
lhd0 = lhs(2, samples=Ninit_GP, criterion='centermaximin')
lhd = uniform(loc=-4,scale=8).ppf(lhd0)
y_HF_LFtrain = np.empty(1, dtype = float)
inp_LFtrain = np.empty([1,2], dtype = float)
for ii in np.arange(0,Ninit_GP,1):
    inp = np.array([lhd[ii,0], lhd[ii,1]]).reshape(2)
    inpp = inp[None, :]
    inp_LFtrain = np.concatenate((inp_LFtrain, inp.reshape(1,2)))
    y_HF_LFtrain = np.concatenate((y_HF_LFtrain, np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)))
inp_LFtrain = np.delete(inp_LFtrain, 0, 0)
y_HF_LFtrain = np.delete(y_HF_LFtrain, 0)
ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,inp_LFtrain), obs = Norm1(y_HF_LFtrain,y_HF_LFtrain), amp_init=1., len_init=1., var_init=1., num_iters = 1000)
amp0, len0, var0 = ML0.GP_train()

Iters = 300
lhd1 = lhs(2, samples=200, criterion='maximin')
lhd =  norm().ppf(lhd1)
y_LF_GP = np.empty(1, dtype = float)
y_HF_GP = np.empty(1, dtype = float)
inp_GPtrain = np.empty([1,2], dtype = float)
Ninit_GP = 12
for ii in np.arange(0,Ninit_GP,1):
    inp = np.array([lhd[ii,0], lhd[ii,1]]).reshape(2)
    inpp = inp[None, :]
    inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,2)))
    samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain), num_samples=num_s)
    y_LF_GP = np.concatenate((y_LF_GP, np.array(np.mean(np.array(samples0),axis=0)).reshape(1)))
    y_HF_GP = np.concatenate((y_HF_GP, np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)))

inp_GPtrain = np.delete(inp_GPtrain, 0, 0)
y_LF_GP = np.delete(y_LF_GP, 0)
y_HF_GP = np.delete(y_HF_GP, 0)

ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain), obs = Norm1((y_HF_GP-y_LF_GP),(y_HF_GP-y_LF_GP)), amp_init=1., len_init=1., var_init=1., num_iters = 1000)
amp1, len1, var1 = ML.GP_train()

## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 2000
Psub = 0.1
Nlim = 3
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,2,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2])

u_GP = np.empty(1, dtype = float)
var_GP = np.empty(1, dtype = float)
var_GP[0] = var1.numpy().reshape(1)
subs_info = np.empty(1, dtype = float)
subs_info[0] = np.array(0).reshape(1)
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)
y_LF_N0 = np.empty(1, dtype = float)
y_LF_N1 = np.empty(1, dtype = float)
y_HF_N1 = np.empty(1, dtype = float)
subs_CV_N0 = np.empty(1, dtype = float)
subs_CV_N1 = np.empty(1, dtype = float)
LF_sto = np.empty(1, dtype = float)

for ii in np.arange(0,Nsub,1):
    inp = DR1.StandardNormal_Indep(N=Ndim)
    inpp = inp[None,:]
    samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain), num_samples=num_s)
    LF = np.array(np.mean(InvNorm1(np.array(samples0),y_HF_LFtrain),axis=0)).reshape(1)
    inp1[ii,:,0] = inp
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain), num_samples=num_s)
    GP_diff = np.mean(InvNorm1(np.array(samples1),(y_HF_GP-y_LF_GP)),axis=0)
    u_check = (np.abs(LF + GP_diff))/np.std(np.array(samples1),axis=0)
    u_GP = np.concatenate((u_GP, u_check))
    u_lim = u_lim_vec[0]
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
        y_LF_N0 = np.concatenate((y_LF_N0, LF))
        subs_CV_N0 = np.concatenate((subs_CV_N0, np.array(0).reshape(1)))
    else:
        y1[ii,0] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,2)))
        y_LF_GP = np.concatenate((y_LF_GP, LF))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        y_LF_N1 = np.concatenate((y_LF_N1, LF))
        y_HF_N1 = np.concatenate((y_HF_N1, y1[ii,0].reshape(1)))
        subs_CV_N1 = np.concatenate((subs_CV_N1, np.array(0).reshape(1)))
        LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
        GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain), obs = Norm1((y_HF_GP-y_LF_GP),(y_HF_GP-y_LF_GP)), amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
        amp1, len1, var1 = ML.GP_train()
        var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
        subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))

count_max = int(Nsub/(Psub*Nsub))

for kk in np.arange(1,Nlim,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
    indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
    inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
    for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
        nxt = np.zeros((1,Ndim))
        
        if count > count_max:
            # ind_max = random.randint(0,int(Psub*Nsub))
            ind_sto = ind_sto + 1
            ind_max = ind_sto
            count = 0
        else:
            ind_max = ii-1
            
        count = count + 1
        
        for jj in np.arange(0,Ndim,1):
            rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
            prop = (rv1.rvs())
            r = rv.pdf((prop))/rv.pdf((inp1[ind_max,jj,kk]))
            if r>uni.rvs():
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ind_max,jj,kk]
            inpp[0,jj] = nxt[0,jj]
        # inpp = inpp[None,:]
        # inpp = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])[None,:]
        samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain), num_samples=num_s)
        LF = np.array(np.mean(InvNorm1(np.array(samples0),y_HF_LFtrain),axis=0)).reshape(1)
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain), num_samples=num_s)
        GP_diff = np.mean(InvNorm1(np.array(samples1),(y_HF_GP-y_LF_GP)),axis=0)
        u_check = (np.abs(LF + GP_diff))/np.std(np.array(samples1),axis=0)
        u_GP = np.concatenate((u_GP, u_check))
        u_lim = u_lim_vec[kk]
        LF_sto = np.concatenate((LF_sto, LF))
        if u_check > u_lim: # and ii > (int(Psub*Nsub)+num_retrain):
            y_nxt = LF + GP_diff
        else:
            y_nxt = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,2)))
            y_LF_GP = np.concatenate((y_LF_GP, LF))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1))) # np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain), obs = Norm1((y_HF_GP-y_LF_GP),(y_HF_GP-y_LF_GP)), amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
            amp1, len1, var1 = ML.GP_train()
            var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
            subs_info = np.concatenate((subs_info, np.array(kk).reshape(1)))
            # GP_diff = 0 ## Comment this
        if (y_nxt)>y1_lim[kk-1] and u_check > u_lim:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
            y_LF_N0 = np.concatenate((y_LF_N0, LF))
            subs_CV_N0 = np.concatenate((subs_CV_N0, np.array(kk).reshape(1)))
        elif (y_nxt)>y1_lim[kk-1] and u_check <= u_lim:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
            y_LF_N1 = np.concatenate((y_LF_N1, LF))
            y_HF_N1 = np.concatenate((y_HF_N1, y_nxt.reshape(1)))
            subs_CV_N1 = np.concatenate((subs_CV_N1, np.array(kk).reshape(1)))
        elif (y_nxt)<=y1_lim[kk-1] and u_GP[len(u_GP)-2] > u_lim:
            inp1[ii,:,kk] = inp1[ind_max,:,kk]
            y1[ii,kk] = y1[ind_max,kk]
            y_LF_N0 = np.concatenate((y_LF_N0, LF_sto[len(LF_sto)-2].reshape(1)))
            subs_CV_N0 = np.concatenate((subs_CV_N0, np.array(kk).reshape(1)))
        else:
            inp1[ii,:,kk] = inp1[ind_max,:,kk]
            y1[ii,kk] = y1[ind_max,kk]
            y_LF_N1 = np.concatenate((y_LF_N1, LF_sto[len(LF_sto)-2].reshape(1)))
            y_HF_N1 = np.concatenate((y_HF_N1, y_HF_GP[len(y_HF_GP)-2].reshape(1)))
            subs_CV_N1 = np.concatenate((subs_CV_N1, np.array(kk).reshape(1)))
 
# Pf = 1
# Pi_sto = np.zeros(Nlim)
# for kk in np.arange(0,Nlim,1):
#     Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(len(inp1[:,0,0]))
#     Pi_sto[kk] = Pi
#     Pf = Pf * Pi

Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)

def f(x):
    return np.float(x)
f2 = np.vectorize(f)
y_LF_N0 = np.delete(y_LF_N0, 0)
subs_CV_N0 = np.delete(subs_CV_N0, 0)
y_LF_N1 = np.delete(y_LF_N1, 0)
y_HF_N1 = np.delete(y_HF_N1, 0)
subs_CV_N1 = np.delete(subs_CV_N1, 0)
Pf_CV = 1.0
Pi_sto_CV = np.zeros(Nlim)
for kk in np.arange(0,Nlim,1):
    ind_N0 = np.where(subs_CV_N0==kk)
    ind_N1 = np.where(subs_CV_N1==kk)
    # Pi = np.sum(f2(y_LF_N0[ind_N0]>np.min([y1_lim[kk],value]))) / len(y_LF_N0[ind_N0]) + np.sum(f2(y_HF_N1[ind_N1]>np.min([y1_lim[kk],value]))) / len(y_HF_N1[ind_N1]) - np.sum(f2(y_LF_N1[ind_N1]>np.min([y1_lim[kk],value]))) / len(y_LF_N1[ind_N1])
    K = y_HF_N1 # f2(y_HF_N1[ind_N1]>np.min([y1_lim[kk],value]))
    K1 = y_LF_N1 # f2(y_LF_N1[ind_N1]>np.min([y1_lim[kk],value]))
    Pi = np.sum(f2(y_HF_N1[ind_N1]>np.min([y1_lim[kk],value]))) / len(y_HF_N1[ind_N1]) + np.corrcoef(K,K1)[0,1] * (np.std(K)/np.std(K1)) * np.sum(f2(y_LF_N0[ind_N0]>np.min([y1_lim[kk],value]))) / len(y_LF_N0[ind_N0])
    Pf_CV = Pf_CV * Pi
    Pi_sto_CV[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))


## Plotting

x = np.arange(-5.0, 6.0, 0.05)
y = np.arange(-5.0, 6.0, 0.05)
X, Y = np.meshgrid(x, y)
Z = np.zeros((len(x),len(y)))
# GP_LF = np.zeros((len(x),len(y)))
for ii in np.arange(0,len(x),1):
    for jj in np.arange(0,len(y),1):
        inp = np.array([x[ii], y[jj]])
        Z[ii,jj] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inp[None,:])))
        # samples1 = ML.GP_predict_mean(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inp[None,:])
        # GP_LF[ii,jj] = np.array(ML.GP_predict_mean(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inp[None,:]))

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
CS.collections[0].set_linewidth(0)
CS.collections[1].set_linewidth(0)
CS.collections[2].set_linewidth(0)
# CS.collections[3].set_linewidth(0)
CS.collections[4].set_linewidth(0)
CS.collections[5].set_linewidth(0)
CS.collections[6].set_linewidth(0)
CS.collections[7].set_linewidth(0)
CS.collections[8].set_linewidth(0)
plt.scatter(inp1[:,0,0],inp1[:,1,0],label='Sub 0')
plt.scatter(inp1[:,0,1],inp1[:,1,1],label='Sub 1')
plt.scatter(inp1[:,0,2],inp1[:,1,2],label='Sub 2')
# plt.scatter(inp1[:,0,3],inp1[:,1,3],label='Sub 3')
# plt.scatter(inp1[:,0,4],inp1[:,1,4],label='Sub 4')
plt.scatter(inp_GPtrain[0:11,0],inp_GPtrain[0:11,1], marker='^', s=100.0,label='HF call (initial)')
plt.scatter(inp_GPtrain[12:1000,0],inp_GPtrain[12:1000,1], marker='^',s=100.0,label='HF call (subsequent)')
plt.legend()
plt.xlim([-5,5])
plt.ylim([-5,5])
plt.xlabel('X1')
plt.ylabel('X2')

# CS.collections[0].set_linewidth(0)
# CS.collections[1].set_linewidth(0)
# CS.collections[2].set_linewidth(0)
# # CS.collections[3].set_linewidth(0)
# CS.collections[4].set_linewidth(0)
# CS.collections[5].set_linewidth(0)
# CS.collections[6].set_linewidth(0)
# CS.collections[7].set_linewidth(0)
# CS.collections[8].set_linewidth(0)

# CS.collections[0].set_linewidth(0)
# CS.collections[1].set_linewidth(0)
# # CS.collections[2].set_linewidth(0)
# CS.collections[3].set_linewidth(0)
# CS.collections[4].set_linewidth(0)
# CS.collections[5].set_linewidth(0)
# CS.collections[6].set_linewidth(0)
# CS.collections[7].set_linewidth(0)