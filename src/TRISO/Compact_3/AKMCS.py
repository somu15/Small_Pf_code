#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:37:56 2021

@author: dhulls
"""

from os import sys
import os
import pathlib
import numpy as np
import random
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import uniform
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle
from statsmodels.distributions.empirical_distribution import ECDF

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
from sklearn.decomposition import PCA

Ndim = 12
value = 0.0

LS1 = LSF()
DR1 = DR()
num_s = 500
P = np.array([213.35e-6,98.9e-6,40.4e-6,35.2e-6,43.4e-6,1,1,1])

## Training GP

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
    return K

rv_out = norm(loc=191023345.13968003, scale=282810513.3103714)
rv_norm = norm(loc=0,scale=1)
jitter_std = 0.1
jitter = norm(loc=0,scale=496994.68*jitter_std)
std_jitter = 0.05

def Norm3(X1,X):
    return (X1-np.mean(X,axis=0))/(np.std(X,axis=0)) # /4031437.0841159956 # /191023345.13968003 #

def InvNorm3(X1,X):
    return (X1*np.std(X,axis=0)+np.mean(X,axis=0)) # *4031437.0841159956 # *191023345.13968003 #


## Train the GP diff model

Ninit_GP = 12
lhd = DR1.StandardNormal_Indep(N=Ninit_GP)
inp_GPtrain = lhd # inp_LFtrain #
y_GPtrain = np.array((LS1.Triso_1d_norm(inp_GPtrain))) # y_HF_LFtrain #
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain)) # Norm1(inp_GPtrain,inp_GPtrain,Ndim)
amp1, len1 = ML.GP_train_kernel(num_iters = 1000, amp_init = 1., len_init = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])) #

## Subset simultion with HF-LF and GP

uni = uniform()
N = 100 # 10000
inpp = DR1.StandardNormal_Indep(N=N)
inp_full = inpp
samples1 = ML.GP_predict_kernel(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
y_GPpred = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
y_GPstd = np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
u_GP = np.abs(y_GPpred)/y_GPstd
u_min = np.min(u_GP)
count = 0
while u_min<2:
    ind_req = np.where(u_GP==u_min)
    new_inp = inpp[ind_req,:]
    count = count + 1
    new_y = (np.array((LS1.Triso_1d_norm(new_inp.reshape(1,Ndim)))).reshape(1))
    inp_GPtrain = np.concatenate((inp_GPtrain, new_inp.reshape(1,Ndim)))
    y_GPtrain = np.concatenate((y_GPtrain, new_y.reshape(1)))
    inpp = np.delete(inpp,ind_req,0)
    ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain)) # Norm1(inp_GPtrain,inp_GPtrain,Ndim)
    amp1, len1 = ML.GP_train_kernel(num_iters = 1000, amp_init = 1., len_init = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])) #
    samples1 = ML.GP_predict_kernel(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
    y_GPpred = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    y_GPstd = np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_GP = np.abs(y_GPpred)/y_GPstd
    u_min = np.min(u_GP)
    print(count)
