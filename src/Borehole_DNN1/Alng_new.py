#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:40:21 2021

@author: dhulls
"""

from os import sys
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
value = 270.0 # 250.0

LS1 = LSF()
DR1 = DR()
num_s = 500
P = np.array([0.1,5.0,52530,120,52.9,120,560,2190])

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        # K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
        K[:,ii] = X1[:,ii]/X[ii]
    return K

def Norm3(X1,X):
    # return ((X1)-np.mean((X)))/(np.std((X)))
    return (X1/300)

def InvNorm3(X1,X):
    # return (X1*np.std((X))+np.mean((X)))
    return (X1*300)

Ninit_GP = 20
lhd = DR1.BoreholeLHS(Ninit_GP)
inp_LFtrain = lhd
y_HF_LFtrain = LS1.Scalar_Borehole_HF_nD(inp_LFtrain)
ML0 = ML_TF(obs_ind = inp_LFtrain, obs = y_HF_LFtrain) # , amp_init=1., len_init=1., var_init=1., num_iters = 1000)
DNN_model = ML0.DNN_train(dim=Ndim, seed=100, neurons1=6, neurons2=4, learning_rate=0.002, epochs=2000)

Ninit_GP = 20
lhd =  DR1.BoreholeRandom(N=Ninit_GP)
inp_GPtrain = lhd
y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain).reshape(Ninit_GP)
y_HF_GP = np.array((LS1.Scalar_Borehole_HF_nD(inp_GPtrain)))
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = 1000)
Iters = 300

uni = uniform()
Nsub = 40000
Psub = 0.1
Nlim = 5
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])

u_GP = np.zeros((Nsub,Nlim))
subs_info = np.zeros((Nsub,Nlim))
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)
additive = value
Indicator = np.ones((Nsub,Nlim))
counter = 1
file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Borehole_DNN1/Results.csv','w')
file1.writelines("0,0,0\n")
file1.close()

# for ii in np.arange(0,100,1):
#     N_doe = 5000
#     inp_doe = DR1.BoreholeRandom(N=N_doe)
#     samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inp_doe,P,Ndim), num_samples=num_s)
#     y_LF_doe = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
#     samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inp_doe,P,Ndim), num_samples=num_s)
#     GP_diff_doe = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
#     u_doe = (np.abs(y_LF_doe + GP_diff_doe - value))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
#     u_min_doe = np.min(u_doe)
#     while u_min_doe<2:
#         # print("Here")
#         ind_doe = np.argmin(u_doe)
#         HF = np.array((LS1.Scalar_Borehole_HF_nD(inp_doe[ind_doe,:][None,:]))).reshape(1)
#         inp_GPtrain = np.concatenate((inp_GPtrain, inp_doe[ind_doe,:].reshape(1,Ndim)))
#         y_LF_GP = np.concatenate((y_LF_GP, y_LF_doe[ind_doe].reshape(1)))
#         y_HF_GP = np.concatenate((y_HF_GP, HF.reshape(1)))
#         y_GPtrain = np.concatenate((y_GPtrain, (HF.reshape(1)-y_LF_doe[ind_doe].reshape(1))))
#         LF_plus_GP = np.concatenate((LF_plus_GP, (y_LF_doe[ind_doe].reshape(1) + GP_diff_doe[ind_doe].reshape(1))))
#         GP_pred = np.concatenate((GP_pred, (GP_diff_doe[ind_doe].reshape(1))))
#         # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
#         ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
#         amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
#         samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inp_doe,P,Ndim), num_samples=num_s)
#         GP_diff_doe = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
#         u_doe = (np.abs(y_LF_doe + GP_diff_doe - value))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
#         u_min_doe = np.min(u_doe)

## Subset simultion with HF-LF and GP

for ii in np.arange(0,Nsub,1):
    inp = DR1.BoreholeRandom().reshape(Ndim)
    inpp = inp[None,:]
    LF = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inpp).reshape(1)
    inp1[ii,:,0] = inp
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    if ii <Ninit_GP:
        additive = np.percentile(y_HF_GP,90)
    else:
        additive = np.percentile(y1[0:ii,0],90)
    u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_GP[ii,0] = u_check

    u_lim = u_lim_vec[0]
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
        y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        y_GPtrain = np.concatenate((y_GPtrain, (y1[ii,0].reshape(1)-LF)))
        LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
        GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
        # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
        amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
        subs_info[ii,0] = 1.0
    file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Borehole_DNN1/Results.csv','r')
    Lines = file1.readlines()
    Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,0])+","+str(subs_info[ii,0])+"\n").reshape(1)))
    file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Borehole_DNN1/Results.csv','w')
    file1.writelines(Lines)
    file1.close()
    counter = counter + 1

LF_plus_GP = np.delete(LF_plus_GP, 0)
GP_pred = np.delete(GP_pred, 0)

inpp = np.zeros((1,Ndim))
count_max = int(Nsub/(Psub*Nsub))-1
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim))
markov_seed = np.zeros(Ndim)
markov_out = 0.0
u_req = np.zeros(Nsub)
u_check1 = 10.0
std_prop = np.zeros(Ndim)

prop_std_req = np.array([0.0216,0.75,11373.07,25.98,11.453,25.98,121.243,474.148])

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
    tmp[:,1:9] = seeds
    np.random.shuffle(tmp)
    seeds_outs = tmp[:,0]
    seeds = tmp[:,1:9]

    for ii in np.arange(0,(Nsub),1):
        print(kk)
        print(ii)
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

            rv1 = norm(loc=np.log(markov_seed[jj]),scale=std_prop[jj])
            prop = np.exp(rv1.rvs())
            r = np.log(DR1.BoreholePDF(rv_req=prop, index=jj)) - np.log(DR1.BoreholePDF(rv_req=(markov_seed[jj]),index=jj))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]
        LF = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inpp).reshape(1)
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
        GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
        if kk<(Nlim-1):
            if ii < Ninit_GP:
                additive =  y1_lim[kk-1]
            else:

                additive = np.percentile(y1[0:ii,kk],90) # y1_lim[kk-1]
        else:
            additive = value
        # additive = y1_lim[kk-1]
        u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
        u_GP[ii,kk] = u_check
        u_lim = u_lim_vec[kk]

        if u_check > u_lim and ii>=Ninit_GP:
            y_nxt = LF + GP_diff
        else:
            y_nxt = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
            y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            y_GPtrain = np.concatenate((y_GPtrain, (y_nxt.reshape(1)-LF)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
            amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
            subs_info[ii,kk] = 1.0

        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = markov_seed
            y1[ii,kk] = markov_out
            Indicator[ii,kk] = 0.0
        file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Borehole_DNN1/Results.csv','r')
        Lines = file1.readlines()
        Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,kk])+","+str(subs_info[ii,kk])+"\n").reshape(1)))
        file1 = open('/home/dhullaks/projects/Small_Pf_code/src/Borehole_DNN1/Results.csv','w')
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

filename = 'Alg_Run1_DNN.pickle'
with open(filename, 'wb') as f:
    pickle.dump(y1, f)
    pickle.dump(y1_lim, f)
    pickle.dump(Pf, f)
    pickle.dump(cov_req, f)
    pickle.dump(Nlim, f)
    pickle.dump(Nsub, f)
    pickle.dump(Pi_sto, f)
    pickle.dump(u_GP, f)
    pickle.dump(y_GPtrain, f)
    pickle.dump(y_HF_GP, f)
    pickle.dump(y_LF_GP, f)
    pickle.dump(subs_info, f)
    pickle.dump(Indicator, f)