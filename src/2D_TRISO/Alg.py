#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 12:18:29 2021

@author: dhulls
"""

from os import sys
import os
import pathlib
import numpy as np
import random
import csv
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import uniform
from scipy.stats import cauchy
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

Ndim = 8
Ndim0 = 6
value = 0.0

LS1 = LSF()
DR1 = DR()
num_s = 500

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


# def Norm3(X1,X):
#     # return ((X1)-np.mean((X)))/(np.std((X)))
#     return X1/(1.2e9)

def Norm3(X1):
    return X1/(1.2e9)

def InvNorm3(X1,X):
    # return (X1*np.std((X))+np.mean((X)))
    return X1

## Train the GP diff model

Iters = 300
Ninit_GP = 12
lhd, lhd0 = DR1.TrisoLHS(Nsamps=Ninit_GP)
y_LF_GP = np.empty(1, dtype = float)
y_HF_GP = np.empty(1, dtype = float)
inp_GPtrain = lhd
inp_GPtrain0 = lhd0
y_LF_GP = Norm3(LS1.Triso_1d(inp_GPtrain0))
y_HF_GP = Norm3(LS1.Triso_2d(inp_GPtrain))
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain0,inp_GPtrain0, Ndim), obs = y_GPtrain)
amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = 1000)

## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 500
Psub = 0.1
Nlim = 4
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim0,Nlim))
inp0 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])

u_GP = np.zeros((Nsub,Nlim))
subs_info = np.zeros((Nsub,Nlim))
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)
additive = value
Indicator = np.ones((Nsub,Nlim))
counter = 1
file1 = open('/home/dhullaks/projects/Small_Pf_code/src/2D_TRISO/Results.csv','w')
file1.writelines("0,0,0\n")
file1.close()

for ii in np.arange(0,Nsub,1):
    inp_HF, inp = DR1.TrisoRandom()
    inpp = inp[None,:]

    LF = Norm3(LS1.Triso_1d(inpp))
    inp0[ii,:,0] = inp
    inp1[ii,:,0] = inp_HF[None,:]

    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain0,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    additive = value
    u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_GP[ii,0] = u_check

    u_lim = u_lim_vec[0]
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = Norm3(np.array((LS1.Triso_2d(inp_HF[None,:]))).reshape(1))
        inp_GPtrain0 = np.concatenate((inp_GPtrain0, inpp.reshape(1,Ndim)))
        y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        y_GPtrain = np.concatenate((y_GPtrain, (y1[ii,0].reshape(1)-LF)))
        LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
        GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
        ML = ML_TF(obs_ind = Norm1(inp_GPtrain0,inp_GPtrain0,Ndim), obs = y_GPtrain)
        amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
        subs_info[ii,0] = 1.0
    file1 = open('/home/dhullaks/projects/Small_Pf_code/src/2D_TRISO/Results.csv','r')
    Lines = file1.readlines()
    Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,0])+","+str(subs_info[ii,0])+"\n").reshape(1)))
    file1 = open('/home/dhullaks/projects/Small_Pf_code/src/2D_TRISO/Results.csv','w')
    file1.writelines(Lines)
    file1.close()
    counter = counter + 1

LF_plus_GP = np.delete(LF_plus_GP, 0)
GP_pred = np.delete(GP_pred, 0)

# inpp = np.zeros((1,Ndim))
count_max = int(Nsub/(Psub*Nsub))-1
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim0))
seeds0 = np.zeros((int(Psub*Nsub),Ndim))
markov_seed = np.zeros(Ndim0)
markov_seed0 = np.zeros(Ndim)
markov_out = 0.0

prop_std = np.array([0.32, 0.32, 0.32, 0.32, 0.32, 0.55, 0.55, 0.55])
ind_HF = [0,1,2,3,4,7]
inpp = np.zeros((1,Ndim))
for kk in np.arange(1,Nlim,1):

    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]
    seeds0 = inp0[indices,:,kk-1]

    for ii in np.arange(0,(Nsub),1):
        nxt = np.zeros((1,Ndim))

        if count > count_max:
            ind_sto = ind_sto + 1
            count = 0
            markov_seed = seeds[ind_sto,:]
            markov_seed0 = seeds0[ind_sto,:]
            markov_out = seeds_outs[ind_sto]
        else:
            markov_seed = inp1[ii-1,:,kk]
            markov_seed0 = inp0[ii-1,:,kk]
            markov_out = y1[ii-1,kk]

        count = count + 1

        for jj in np.arange(0,Ndim,1):
            rv1 = norm(loc=np.log(markov_seed0[jj]),scale=prop_std[jj])
            prop = np.exp(rv1.rvs())
            r = np.log(DR1.TrisoPDF(rv_req=prop, index=jj)) - np.log(DR1.TrisoPDF(rv_req=(markov_seed0[jj]),index=jj)) # np.log(rv.pdf((prop)))-np.log(rv.pdf((inp1[ind_max,jj,kk])))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed0[jj]
            inpp[0,jj] = nxt[0,jj]
        LF = Norm3(LS1.Triso_1d(inpp.reshape(1,Ndim)))
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp.reshape(1,Ndim),inp_GPtrain0,Ndim), num_samples=num_s)
        GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
        additive = y1_lim[kk-1]
        u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)

        u_GP[ii,kk] = u_check
        u_lim = u_lim_vec[kk]

        print(ii)
        print(kk)
        if u_check > u_lim:
            y_nxt = LF + GP_diff
        else:
            y_nxt = Norm3(np.array((LS1.Triso_2d(inpp[0,ind_HF].reshape(1,Ndim0)))).reshape(1))
            inp_GPtrain0 = np.concatenate((inp_GPtrain0, inpp.reshape(1,Ndim)))
            y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            y_GPtrain = np.concatenate((y_GPtrain, (y_nxt.reshape(1)-LF)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain0,inp_GPtrain0,Ndim), obs = y_GPtrain)
            amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
            subs_info[ii,kk] = 1.0

        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp[0,ind_HF]
            inp0[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = markov_seed
            inp0[ii,:,kk] = markov_seed0
            y1[ii,kk] = markov_out
            Indicator[ii,kk] = 0.0
        file1 = open('/home/dhullaks/projects/Small_Pf_code/src/2D_TRISO/Results.csv','r')
        # file1 = open('/home/dhullaks/projects/bison/examples/TRISO/2D_Alg/2d_MC.i', 'r')
        Lines = file1.readlines()
        Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,kk])+","+str(subs_info[ii,kk])+"\n").reshape(1)))
        file1 = open('/home/dhullaks/projects/Small_Pf_code/src/2D_TRISO/Results.csv','w')
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

filename = 'Alg_Run.pickle'
os.chdir('/home/dhullaks/projects/Small_Pf_code/src/2D_TRISO')
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
    pickle.dump(y_GPtrain, f)
    pickle.dump(y_HF_GP, f)
    pickle.dump(y_LF_GP, f)
    pickle.dump(inp_GPtrain0, f)
    pickle.dump(Indicator, f)

# # req = 2.425e-04 (1000 sims per subset and 4 subsets; Num HF = 86; value=800)

# # plt.scatter(y_HF_GP[12:86],LF_plus_GP)
# # plt.xlim([700,900])
