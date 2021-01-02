#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:43:49 2020

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
from pyDOE import *

def Convert(lst): 
    return [ -i for i in lst ] 

## Monte Carlo simulations

# Nsims = 250000
# y = np.zeros(Nsims)
# ys = np.zeros(Nsims)
# LS1 = LSF()
# DR1 = DR()
# Ndim = 2
# value = 0.25

# for ii in np.arange(0,Nsims,1):
#     inp = (DR1.StandardNormal_Indep(N=Ndim))
#     inpp = inp[None,:]
#     y[ii] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp)))
#     ys[ii] = np.array(Convert(LS1.Scalar_LS1_LF_2D(inpp)))

# req = len(np.rot90(np.where(y>value)))/Nsims

# req = 0.0022256
# corr = 0.353


## Visualize limit state

# req = np.arange(-5,5,0.05)
# req_y = np.zeros((len(x),len(x)))
# req_y1 = np.zeros((len(x),len(x)))

# for ii in np.arange(0,len(req),1):
#     for jj in np.arange(0,len(req),1):
#         req_y[ii,jj] = LS1.Scalar_LS1_LF_2D(np.array([req[ii],req[jj]]).reshape(1,2))
#         req_y1[ii,jj] = LS1.Scalar_LS1_HF_2D(np.array([req[ii],req[jj]]).reshape(1,2))

# X, Y = np.meshgrid(req, req)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, req_y1, rstride=1, cstride=1, cmap='summer', edgecolor='none')
# ax.set_title('High fidelity');

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, req_y, rstride=1, cstride=1, cmap='winter',edgecolor='none')
# ax.set_title('Low fidelity');


# ax = plt.axes(projection="3d")
# ax.plot_wireframe(X, Y, y, color='green')


Ndim = 2
LS1 = LSF()
DR1 = DR()
num_s = 500
value = 0.25 
Iters = 250

## Training GP

# uniform(loc=-5,scale=10).rvs()

lhd = lhs(2, samples=200, criterion='maximin')
lhd = uniform(loc=-5,scale=10).ppf(lhd)
y_LF_GP = np.empty(1, dtype = float)
y_HF_GP = np.empty(1, dtype = float)
inp_GPtrain = np.empty([1,2], dtype = float)
Ninit_GP = 50
for ii in np.arange(0,Ninit_GP,1):
    inp = np.array([lhd[ii,0], lhd[ii,1]]).reshape(2)
    inpp = inp[None, :]
    inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,2)))
    y_LF_GP = np.concatenate((y_LF_GP, np.array(Convert(LS1.Scalar_LS1_LF_2D(inpp))).reshape(1)))
    y_HF_GP = np.concatenate((y_HF_GP, np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)))

inp_GPtrain = np.delete(inp_GPtrain, 0, 0)
y_LF_GP = np.delete(y_LF_GP, 0)
y_HF_GP = np.delete(y_HF_GP, 0)

ML = ML_TF(obs_ind = inp_GPtrain, obs = (y_HF_GP-y_LF_GP), amp_init=1., len_init=1., var_init=1., num_iters = 1000)
amp1, len1, var1 = ML.GP_train()
# samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inp_GPtrain[:,None], num_samples=num_s)
x_req = np.array(lhd[np.arange((Ninit_GP+1),200,1),:]).reshape(len(np.array(lhd[np.arange((Ninit_GP+1),200,1),0])),2)
samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = x_req, num_samples=num_s)
LF_req = np.array(Convert(LS1.Scalar_LS1_LF_2D(x_req)))#.reshape(1)
u_req = (np.abs(LF_req + np.mean(np.array(samples1),axis=0)))/np.std(np.array(samples1),axis=0)
HF_req = np.array(Convert(LS1.Scalar_LS1_HF_2D(x_req)))
ind_req = np.rot90(np.where(u_req<2))

for ii in np.arange(0,len(ind_req),1):
    inp = np.array([lhd[ii,0], lhd[ii,1]]).reshape(2)
    inpp = inp[None, :]
    inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,2)))
    y_LF_GP = np.concatenate((y_LF_GP, np.array(Convert(LS1.Scalar_LS1_LF_2D(inpp))).reshape(1)))
    y_HF_GP = np.concatenate((y_HF_GP, np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)))
    ML = ML_TF(obs_ind = inp_GPtrain, obs = (y_HF_GP-y_LF_GP), amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
    amp1, len1, var1 = ML.GP_train()


## Subset simultion with HF-LF and GP

# Ninit_GP = 50
# y_LF_GP = np.empty(1, dtype = float)
# y_HF_GP = np.empty(1, dtype = float)
# inp_GPtrain = np.empty(1, dtype = float)
# for ii in np.arange(0,Ninit_GP,1):
#     inp = (DR1.StandardNormal_Indep(N=Ndim))
#     inpp = inp[None, :]
#     inp_GPtrain = np.concatenate((inp_GPtrain, inp))
#     y_LF_GP = np.concatenate((y_LF_GP, LS1.Scalar_LS2_LF(inpp)))
#     y_HF_GP = np.concatenate((y_HF_GP, LS1.Scalar_LS2_HF(inpp)))

# ML = ML_TF(obs_ind = inp_GPtrain[:,None], obs = (y_HF_GP-y_LF_GP))
# amp1, len1, var1 = ML.GP_train()
# samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inp_GPtrain[:,None], num_samples=num_s)

uni = uniform()
Nsub = 2500
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

for ii in np.arange(0,Nsub,1):
    inp = DR1.StandardNormal_Indep(N=Ndim)
    inpp = inp[None,:]
    LF = np.array(Convert(LS1.Scalar_LS1_LF_2D(inpp))).reshape(1)
    inp1[ii,:,0] = inp
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inpp, num_samples=num_s)
    GP_diff = np.mean(np.array(samples1),axis=0)
    u_check = (np.abs(LF + GP_diff))/np.std(np.array(samples1),axis=0)
    u_GP = np.concatenate((u_GP, u_check))
    u_lim = u_lim_vec[0]
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,2)))
        y_LF_GP = np.concatenate((y_LF_GP, LF))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
        GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
        # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
        ML = ML_TF(obs_ind = inp_GPtrain, obs = (y_HF_GP-y_LF_GP), amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
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
            inpp[0,jj] = nxt[0,jj]
        # inpp = inpp[None,:]
        # inpp = np.array([nxt[0,0], nxt[0,1], nxt[0,2]])[None,:]
        LF = np.array(Convert(LS1.Scalar_LS1_LF_2D(inpp))).reshape(1)
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inpp, num_samples=num_s)
        GP_diff = np.mean(np.array(samples1),axis=0)
        u_check = (np.abs(LF + GP_diff))/np.std(np.array(samples1),axis=0)
        u_GP = np.concatenate((u_GP, u_check))
        u_lim = u_lim_vec[kk]
        if u_check > u_lim:
            y_nxt = LF + GP_diff
        else:
            y_nxt = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,2)))
            y_LF_GP = np.concatenate((y_LF_GP, LF))
            y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            ML = ML_TF(obs_ind = inp_GPtrain, obs = (y_HF_GP-y_LF_GP), amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
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
