#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 15:35:40 2020

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

Ndim = 10
value = 0.14

LS1 = LSF()
DR1 = DR()
num_s = 500

## Monte Carlo simulations

# Nsims = int(4e6)
# y = np.zeros(Nsims)
# ys = np.zeros(Nsims)
# LS1 = LSF()
# DR1 = DR()
# Ndim = 10
# value = 0.14

# for ii in np.arange(0,Nsims,1):
#     inp = (DR1.TrussRandom())
#     inpp = inp[None,:]
#     y[ii] = np.array(LS1.Truss_HF(inpp))
#     print(ii/Nsims)

# req = len(np.rot90(np.where(y>value)))/Nsims

# req = 0.0001469 (2.4e6 simulations)

# Basic subset simulation

# LS1 = LSF()
# DR1 = DR()
# num_s = 500

# uni = uniform()
# Nsub = 6000
# Psub = 0.1
# Nlim = 5
# y1 = np.zeros((Nsub,Nlim))
# y1_lim = np.zeros(Nlim)
# # y1_lim[Nlim-1] = value
# inp1 = np.zeros((Nsub,Ndim,Nlim))
# rv = norm(loc=0,scale=1)
# y_seed = np.zeros(int(Psub*Nsub))

# for ii in np.arange(0,Nsub,1):
#     inp = (DR1.BoreholeRandom())
#     inpp = inp[None,:]
#     y1[ii,0] = np.array(LS1.Scalar_Borehole_HF_nD(inpp))
#     inp1[ii,:,0] = inp

# inpp = np.zeros(Ndim)
# count_max = Nsub/(Psub*Nsub)
# count = 100000
# ind_max = 1
# r_sto = np.zeros((Nsub-int(Psub*Nsub),Nlim-1,Ndim))
# # ind_sto = 3

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
#             if jj == 0:
#                 rv1 = norm(loc=inp1[ind_max,jj,kk],scale=0.1)
#             else:
#                 rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
#             prop = (rv1.rvs())
#             r = np.log(DR1.BoreholePDF(rv_req=prop, index=jj)) - np.log(DR1.BoreholePDF(rv_req=(inp1[ind_max,jj,kk]),index=jj)) # rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
#             r_sto[ii-(int(Psub*Nsub)),kk-1,jj] = r
#             if r>np.log(uni.rvs()):
#                 nxt[0,jj] = prop
#             else: 
#                 nxt[0,jj] = inp1[ind_max,jj,kk]
#             inpp[jj] = nxt[0,jj]
#         y_nxt = np.array(LS1.Scalar_Borehole_HF_nD(inpp[None,:])).reshape(1)
#         if y_nxt>y1_lim[kk-1]:
#             inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
#             y1[ii,kk] = y_nxt
#         else:
#             inp1[ii,:,kk] = inp1[ind_max,:,kk]
#             y1[ii,kk] = y1[ind_max,kk]

# Pf = 1
# Pi_sto = np.zeros(Nlim)
# cov_sq = 0
# for kk in np.arange(0,Nlim,1):
#     Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
#     Pf = Pf * Pi
#     Pi_sto[kk] = Pi
#     cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
# cov_req = np.sqrt(cov_sq)

# # Pf = 0.0003651 [cov_req = 0.0437; 5 subsets with 15000 sims per subset]

## Training GP

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = np.reshape((np.log(X1[:,ii])-np.mean(np.log(X[:,ii])))/(np.std(np.log(X[:,ii]))),len(X1))
    return K

def Norm2(X1,X):
    return (np.log(X1)-np.mean(np.log(X)))/(np.std(np.log(X)))

# def InvNorm1(X1,X):
#     return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

def InvNorm2(X1,X):
    return np.exp(X1*np.std(np.log(X))+np.mean(np.log(X)))


def Norm3(X1,X):
    return ((X1)-np.mean((X)))/(np.std((X)))

def InvNorm3(X1,X):
    return (X1*np.std((X))+np.mean((X)))

# Ninit_GP = 50
# lhd = DR1.BoreholeLHS(Ninit_GP) #  uniform(loc=-3.5,scale=7.0).ppf(lhd0) # 
# inp_LFtrain = lhd
# y_HF_LFtrain = LS1.Scalar_Borehole_HF_nD(inp_LFtrain)
# ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,inp_LFtrain,Ndim), obs = Norm2(y_HF_LFtrain,y_HF_LFtrain), amp_init=1.0, len_init=1.0, var_init=1.0, num_iters = 10000)
# amp0, len0, var0 = ML0.GP_train()

Ninit_GP = 12
lhd = DR1.TrussLHS(Ninit_GP) #  uniform(loc=-3.5,scale=7.0).ppf(lhd0) # 
inp_LFtrain = lhd
y_HF_LFtrain = LS1.Truss_HF(inp_LFtrain)
ML0 = ML_TF(obs_ind = inp_LFtrain, obs = y_HF_LFtrain) # , amp_init=1., len_init=1., var_init=1., num_iters = 1000)
DNN_model = ML0.DNN_train(dim=Ndim, seed=100, neurons1=6, neurons2=4, learning_rate=0.002, epochs=5000)

## Train the GP diff model

Iters = 300
Ninit_GP = 12
lhd = DR1.TrussLHS(Nsamps=Ninit_GP)
y_LF_GP = np.empty(1, dtype = float)
y_HF_GP = np.empty(1, dtype = float)
inp_GPtrain = lhd
y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0]
y_HF_GP = LS1.Truss_HF(inp_GPtrain)
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain, Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
amp1, len1, var1 = ML.GP_train(amp_init=1., len_init=1., var_init=1., num_iters = 1000)

# Ninit_GP = 50
# lhd =  DR1.TrussLHS(Nsamps=Ninit_GP)
# inp_GPtrain = lhd
# samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inp_GPtrain,inp_LFtrain,Ndim), num_samples=num_s)
# y_LF_GP = np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
# y_HF_GP = np.array((LS1.Scalar_Borehole_HF_nD(inp_GPtrain)))
# # std_check = np.std(InvNorm2(np.array(samples0),y_HF_LFtrain),axis=0)
# y_GPtrain = y_HF_GP - y_LF_GP
# ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain), amp_init=1., len_init=1., var_init=1., num_iters = 1000)
# amp1, len1, var1 = ML.GP_train()
# Iters = 300

# Ninit_GP = 500
# lhd =  DR1.BoreholeLHS(Nsamps=Ninit_GP)
# inp_GPtrain1 = lhd
# samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inp_GPtrain1,inp_LFtrain,Ndim), num_samples=num_s)
# y_LF_GP = np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
# y_HF_GP = np.array((LS1.Scalar_Borehole_HF_nD(inp_GPtrain1)))
# samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inp_GPtrain1,inp_GPtrain1,Ndim), num_samples=num_s)
# K = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
# std_check = np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)

## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 5000
Psub = 0.1
Nlim = 4
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])

u_GP = np.empty(1, dtype = float)
var_GP = np.empty(1, dtype = float)
std_GPdiff = np.empty(1, dtype = float)
var_GP[0] = var1.numpy().reshape(1)
subs_info = np.empty(1, dtype = float)
subs_info[0] = np.array(0).reshape(1)
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)

for ii in np.arange(0,Nsub,1):
    inp = DR1.TrussRandom()
    inpp = inp[None,:]
    # samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain,Ndim), num_samples=num_s)
    # LF = np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
    LF = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inpp)[0]
    inp1[ii,:,0] = inp
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    u_check = (np.abs(LF + GP_diff - value))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
    std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)).reshape(1)))
    u_lim = u_lim_vec[0]
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = np.array((LS1.Truss_HF(inpp))).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
        y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        y_GPtrain = np.concatenate((y_GPtrain, (y1[ii,0].reshape(1)-LF)))
        LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
        GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
        # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
        amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
        var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
        subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))
        
u_GP = np.delete(u_GP, 0)
var_GP = np.delete(var_GP, 0)
std_GPdiff = np.delete(std_GPdiff, 0)
subs_info = np.delete(subs_info, 0)
LF_plus_GP = np.delete(LF_plus_GP, 0)
GP_pred = np.delete(GP_pred, 0)

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
            rv1 = norm(loc=np.log(inp1[ind_max,jj,kk]),scale=1.0)
            prop = np.exp(rv1.rvs())
            r = np.log(DR1.TrussPDF(rv_req=prop, index=jj)) - np.log(DR1.TrussPDF(rv_req=(inp1[ind_max,jj,kk]),index=jj)) # np.log(rv.pdf((prop)))-np.log(rv.pdf((inp1[ind_max,jj,kk])))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ind_max,jj,kk]
            inpp[0,jj] = nxt[0,jj]
        # samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain,Ndim), num_samples=num_s)
        # LF = np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
        LF = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inpp)[0]
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
        GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
        u_check = (np.abs(LF + GP_diff - value))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
        u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
        std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)).reshape(1)))
        u_lim = u_lim_vec[kk]
        if u_check > u_lim: # and ii > (int(Psub*Nsub)+num_retrain):
            y_nxt = LF + GP_diff
        else:
            y_nxt = np.array((LS1.Truss_HF(inpp))).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
            y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            y_GPtrain = np.concatenate((y_GPtrain, (y_nxt.reshape(1)-LF)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
            amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
            var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
            subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))
            
        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
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

# 