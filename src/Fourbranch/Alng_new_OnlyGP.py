#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 22:24:19 2022

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
import csv
import matplotlib.pyplot as plt
import pickle

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

LS1 = LSF()
DR1 = DR()
num_s = 500

## Training GP

def Norm1(X1,X):
    return X1 # (X1-np.mean(X,axis=0))/(np.std(X,axis=0))

def InvNorm1(X1,X):
    return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

Ninit_GP = 20
lhd0 = lhs(2, samples=Ninit_GP, criterion='centermaximin')
lhd = norm().ppf(lhd0) # uniform(loc=-3,scale=6).ppf(lhd0)
y_HF_LFtrain = np.empty(1, dtype = float)
inp_LFtrain = np.empty([1,2], dtype = float)
for ii in np.arange(0,Ninit_GP,1):
    inp = np.array([lhd[ii,0], lhd[ii,1]]).reshape(2)
    inpp = inp[None, :]
    inp_LFtrain = np.concatenate((inp_LFtrain, inp.reshape(1,2)))
    y_HF_LFtrain = np.concatenate((y_HF_LFtrain, np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)))
inp_LFtrain = np.delete(inp_LFtrain, 0, 0)
y_HF_LFtrain = np.delete(y_HF_LFtrain, 0)
ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,inp_LFtrain), obs = Norm1(y_HF_LFtrain,y_HF_LFtrain))
amp0, len0 = ML0.GP_train(amp_init=1., len_init=1., num_iters = 1000)

## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 20000
Psub = 0.1
Nlim = 3
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,2,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2])

u_GP = np.zeros((Nsub,Nlim))
subs_info = np.zeros((Nsub,Nlim))
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)
additive = value
Indicator = np.ones((Nsub,Nlim))
counter = 1
file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Fourbranch/Results.csv','w')
file1.writelines("0,0,0\n")
file1.close()
y_sto_all = np.zeros((Nsub,Nlim))

for ii in np.arange(0,Nsub,1):
    inp = DR1.StandardNormal_Indep(N=Ndim)
    inpp = inp[None,:]
    LF = ML0.GP_predict_mean(amplitude_var = amp0, length_scale_var=len0, pred_ind = inpp).reshape(1)
    inp1[ii,:,0] = inp
    if ii <Ninit_GP:
        additive = np.percentile(y_HF_LFtrain,90)
    else:
        additive = np.percentile(y1[0:ii,0],90)
    u_check = (np.abs(LF - additive))/ML0.GP_predict_std(amplitude_var = amp0, length_scale_var=len0, pred_ind = inpp).reshape(1)
    u_GP[ii,0] = u_check
    u_lim = u_lim_vec[0]
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF 
    else:
        y1[ii,0] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
        inp_LFtrain = np.concatenate((inp_LFtrain, inpp.reshape(1,2)))
        y_HF_LFtrain = np.concatenate((y_HF_LFtrain, y1[ii,0].reshape(1)))
        ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,inp_LFtrain), obs = Norm1(y_HF_LFtrain,y_HF_LFtrain))
        amp0, len0 = ML0.GP_train(amp_init=1., len_init=1., num_iters = 1000)
        subs_info[ii,0] = 1.0
    file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Fourbranch/Results.csv','r')
    Lines = file1.readlines()
    Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,0])+","+str(subs_info[ii,0])+"\n").reshape(1)))
    file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Fourbranch/Results.csv','w')
    file1.writelines(Lines)
    file1.close()
    counter = counter + 1

inpp = np.zeros((1,Ndim))
count_max = int(Nsub/(Psub*Nsub))-1
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim))
markov_seed = np.zeros(Ndim)
markov_out = 0.0
u_req = np.zeros(Nsub)
u_check1 = 10.0

for kk in np.arange(1,Nlim,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]
    std_prop = np.log((seeds)).std(0)
    tmp = np.zeros((len(seeds_outs),1+Ndim))
    tmp[:,0] = seeds_outs
    tmp[:,1:3] = seeds
    np.random.shuffle(tmp)
    seeds_outs = tmp[:,0]
    seeds = tmp[:,1:3]

    for ii in np.arange(0,(Nsub),1):
        nxt = np.zeros((1,Ndim))

        if count > count_max:
            ind_sto = ind_sto + 1
            count = 0
            markov_seed = seeds[ind_sto,:]
            markov_out = seeds_outs[ind_sto]
        else:
            markov_seed = inp1[ii-1,:,kk]
            markov_out = y1[ii-1,kk]

        count = count + 1

        for jj in np.arange(0,Ndim,1):
            rv1 = norm(loc=markov_seed[jj],scale=1.0)
            prop = (rv1.rvs())
            r = rv.pdf((prop))/rv.pdf((markov_seed[jj]))
            if r>uni.rvs():
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]
        LF = ML0.GP_predict_mean(amplitude_var = amp0, length_scale_var=len0, pred_ind = inpp).reshape(1)
        if kk<(Nlim-1):
            if ii < Ninit_GP:
                additive =  y1_lim[kk-1]
            else:

                additive = np.percentile(y1[0:ii,kk],90)
        else:
            additive = value
        u_check = (np.abs(LF - additive))/ML0.GP_predict_std(amplitude_var = amp0, length_scale_var=len0, pred_ind = inpp).reshape(1)

        u_GP[ii,kk] = u_check
        u_lim = u_lim_vec[kk]

        print(ii)
        print(kk)
        if u_check >= u_lim and ii>=Ninit_GP:
            y_nxt = LF 
        else:
            y_nxt = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
            inp_LFtrain = np.concatenate((inp_LFtrain, inpp.reshape(1,2)))
            y_HF_LFtrain = np.concatenate((y_HF_LFtrain, y_nxt.reshape(1)))
            ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,inp_LFtrain), obs = Norm1(y_HF_LFtrain,y_HF_LFtrain))
            amp0, len0 = ML0.GP_train(amp_init=1., len_init=1., num_iters = 1000)
            subs_info[ii,kk] = 1.0
        y_sto_all[ii,kk] = y_nxt
        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = markov_seed
            y1[ii,kk] = markov_out
            Indicator[ii,kk] = 0.0
        file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Fourbranch/Results.csv','r')
        Lines = file1.readlines()
        Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,kk])+","+str(subs_info[ii,kk])+"\n").reshape(1)))
        file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Fourbranch/Results.csv','w')
        file1.writelines(Lines)
        file1.close()
        counter = counter + 1

Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)

filename = 'Alg_Run1_OnlyGP.pickle'
with open(filename, 'wb') as f:
    pickle.dump(y1, f)
    pickle.dump(y1_lim, f)
    pickle.dump(Pf, f)
    pickle.dump(cov_req, f)
    pickle.dump(Nlim, f)
    pickle.dump(Nsub, f)
    pickle.dump(Pi_sto, f)
    pickle.dump(u_GP, f)
    pickle.dump(subs_info, f)
    pickle.dump(Indicator, f)
    pickle.dump(inp_LFtrain, f)
    pickle.dump(y_HF_LFtrain, f)
