#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:10:37 2020

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

Ndim = 8
value = 250

LS1 = LSF()
DR1 = DR()
num_s = 500

## Training GP

def Norm1(X1,X):
    return (X1-np.mean(X,axis=0))/(np.std(X,axis=0))

def Norm2(X1,X):
    return (X1-np.mean(X,axis=0))/(np.std(X,axis=0))

# def InvNorm1(X1,X):
#     return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

def InvNorm2(X1,X):
    return (X1*np.std(X,axis=0)+np.mean(X,axis=0))

Ninit_GP = 50
lhd = DR1.BoreholeLHS(Ninit_GP) #  uniform(loc=-3.5,scale=7.0).ppf(lhd0) # 
inp_LFtrain = lhd
y_HF_LFtrain = LS1.Scalar_Borehole_HF_nD(inp_LFtrain)
ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,inp_LFtrain), obs = Norm2(y_HF_LFtrain,y_HF_LFtrain), amp_init=1.0, len_init=1.0, var_init=1.0, num_iters = 1000)
amp0, len0, var0 = ML0.GP_train()

Ninit_GP = 12
lhd =  DR1.BoreholeLHS(Ninit_GP)
inp_GPtrain = lhd
samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inp_GPtrain,inp_LFtrain), num_samples=num_s)
y_LF_GP = np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
y_HF_GP = np.array((LS1.Scalar_Borehole_HF_nD(inp_GPtrain)))
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain), obs = y_GPtrain, amp_init=1., len_init=1., var_init=1., num_iters = 1000)
amp1, len1, var1 = ML.GP_train()
Iters = 300

# y_HF_LFtrain = np.empty(1, dtype = float)
# inp_LFtrain = np.empty([1,Ndim], dtype = float)
# for ii in np.arange(0,Ninit_GP,1):
#     inp = lhd[ii,:].reshape(Ndim)
#     inpp = inp[None, :]
#     inp_LFtrain = np.concatenate((inp_LFtrain, inp.reshape(1,Ndim)))
#     y_HF_LFtrain = np.concatenate((y_HF_LFtrain, np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)))
# inp_LFtrain = np.delete(inp_LFtrain, 0, 0)
# y_HF_LFtrain = np.delete(y_HF_LFtrain, 0)

# Iters = 300
# lhd =  DR1.BoreholeLHS(200)
# y_LF_GP = np.empty(1, dtype = float)
# y_HF_GP = np.empty(1, dtype = float)
# inp_GPtrain = np.empty([1,Ndim], dtype = float)
# y_GPtrain = np.empty(1, dtype = float)
# Ninit_GP = 12
# for ii in np.arange(0,Ninit_GP,1):
#     inp = lhd[ii,:].reshape(Ndim)
#     inpp = inp[None, :]
#     inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
#     samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain), num_samples=num_s)
#     y_LF_GP = np.concatenate((y_LF_GP, np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain)).reshape(1)))
#     y_HF_GP = np.concatenate((y_HF_GP, np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)))
#     y_GPtrain = np.concatenate((y_GPtrain, (np.array((LS1.Scalar_Borehole_HF_nD(inpp))-np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain)))).reshape(1)))

# inp_GPtrain = np.delete(inp_GPtrain, 0, 0)
# y_LF_GP = np.delete(y_LF_GP, 0)
# y_HF_GP = np.delete(y_HF_GP, 0)
# y_GPtrain = np.delete(y_GPtrain, 0)

# y_GPtrain = (y_HF_GP-y_LF_GP)
# ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain), obs = y_GPtrain, amp_init=1., len_init=1., var_init=1., num_iters = 1000)
# amp1, len1, var1 = ML.GP_train()

## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 250
Psub = 0.1
Nlim = 5
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])

u_GP = np.empty(1, dtype = float)
var_GP = np.empty(1, dtype = float)
var_GP[0] = var1.numpy().reshape(1)
subs_info = np.empty(1, dtype = float)
subs_info[0] = np.array(0).reshape(1)
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)

for ii in np.arange(0,Nsub,1):
    inp = DR1.BoreholeRandom()
    inpp = inp[None,:]
    samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain), num_samples=num_s)
    LF = np.array(np.mean(InvNorm2(np.array(samples0),y_HF_LFtrain),axis=0)).reshape(1)
    inp1[ii,:,0] = inp
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain), num_samples=num_s)
    GP_diff = np.mean((np.array(samples1)),axis=0)
    u_check = (np.abs(LF + GP_diff))/np.std(np.array(samples1),axis=0)
    u_GP = np.concatenate((u_GP, u_check))
    u_lim = u_lim_vec[0]
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
        y_LF_GP = np.concatenate((y_LF_GP, LF))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        y_GPtrain = np.concatenate((y_GPtrain, (y1[ii,0].reshape(1)-LF)))
        LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
        GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
        # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain), obs = y_GPtrain, amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
        amp1, len1, var1 = ML.GP_train()
        var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
        subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))

# inpp = np.zeros(Ndim)
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
            if jj == 0:
                rv1 = norm(loc=inp1[ind_max,jj,kk],scale=0.1)
            else:
                rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
            # rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
            prop = (rv1.rvs())
            r = np.log(DR1.BoreholePDF(rv_req=prop, index=jj)) - np.log(DR1.BoreholePDF(rv_req=(inp1[ind_max,jj,kk]),index=jj)) # np.log(rv.pdf((prop)))-np.log(rv.pdf((inp1[ind_max,jj,kk])))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ind_max,jj,kk]
            inpp[0,jj] = nxt[0,jj]
        samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain), num_samples=num_s)
        LF = np.array(np.mean(InvNorm2(np.array(samples0),y_HF_LFtrain),axis=0)).reshape(1)
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain), num_samples=num_s)
        GP_diff = np.mean((np.array(samples1)),axis=0)
        u_check = (np.abs(LF + GP_diff))/np.std(np.array(samples1),axis=0)
        u_GP = np.concatenate((u_GP, u_check))
        u_lim = u_lim_vec[kk]
        if u_check > u_lim: # and ii > (int(Psub*Nsub)+num_retrain):
            y_nxt = LF + GP_diff
        else:
            # y_nxt = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
            # inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
            # y_LF_GP = np.concatenate((y_LF_GP, LF))
            # y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
            # LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            # GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            # ML = ML_TF(obs_ind = inp_GPtrain, obs = (y_HF_GP-y_LF_GP), amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
            # amp1, len1, var1 = ML.GP_train()
            # var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
            # subs_info = np.concatenate((subs_info, np.array(kk).reshape(1)))
            # GP_diff = 0 ## Comment this
            y_nxt = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
            y_LF_GP = np.concatenate((y_LF_GP, LF))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            y_GPtrain = np.concatenate((y_GPtrain, (y_nxt.reshape(1)-LF)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain), obs = y_GPtrain, amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
            amp1, len1, var1 = ML.GP_train()
            var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
            subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))
        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = inp1[ind_max,:,kk]
            y1[ii,kk] = y1[ind_max,kk]
      
# for kk in np.arange(1,Nlim,1):
#     count = np.inf
#     ind_max = 0
#     ind_sto = -1
#     y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
#     y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
#     indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
#     inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
#     for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
#         nxt = np.zeros((1,Ndim))
        
#         if count > count_max:
#             # ind_max = random.randint(0,int(Psub*Nsub))
#             ind_sto = ind_sto + 1
#             ind_max = ind_sto
#             count = 0
#         else:
#             ind_max = ii-1
            
#         count = count + 1
        
#         for jj in np.arange(0,Ndim,1):
#             if jj == 0:
#                 rv1 = norm(loc=inp1[ind_max,jj,kk],scale=0.1)
#             else:
#                 rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
#             prop = (rv1.rvs())
#             r = np.log(DR1.BoreholePDF(rv_req=prop, index=jj)) - np.log(DR1.BoreholePDF(rv_req=(inp1[ind_max,jj,kk]),index=jj)) # rv.pdf((prop))/rv.pdf((inp1[ind_max,jj,kk]))
#             if r>np.log(uni.rvs()):
#                 nxt[0,jj] = prop
#             else: 
#                 nxt[0,jj] = inp1[ind_max,jj,kk]
#             inpp[0,jj] = nxt[0,jj]
#         # inpp = inpp[None,:]
#         # inpp = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])[None,:]
#         samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain), num_samples=num_s)
#         LF = np.array(np.mean((np.array(samples0)),axis=0)).reshape(1)
#         samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain), num_samples=num_s)
#         GP_diff = np.mean((np.array(samples1)),axis=0)
#         u_check = (np.abs(LF + GP_diff))/np.std(np.array(samples1),axis=0)
#         u_GP = np.concatenate((u_GP, u_check))
#         u_lim = u_lim_vec[kk]
#         if u_check > u_lim: # and ii > (int(Psub*Nsub)+num_retrain):
#             y_nxt = LF # + GP_diff
#         else:
#             y_nxt = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
#             inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
#             y_LF_GP = np.concatenate((y_LF_GP, LF))
#             y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1))) # np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
#             LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
#             GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
#             ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain), obs = (y_HF_GP-y_LF_GP), amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
#             amp1, len1, var1 = ML.GP_train()
#             var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
#             subs_info = np.concatenate((subs_info, np.array(kk).reshape(1)))
#             # GP_diff = 0 ## Comment this
#         if (y_nxt)>y1_lim[kk-1]:
#             inp1[ii,:,kk] = inpp
#             y1[ii,kk] = y_nxt
#         else:
#             inp1[ii,:,kk] = inp1[ind_max,:,kk]
#             y1[ii,kk] = y1[ind_max,kk]

Pf = 1
Pi_sto = np.zeros(Nlim)
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(len(inp1[:,0,0]))
    Pi_sto[kk] = Pi
    Pf = Pf * Pi
    


