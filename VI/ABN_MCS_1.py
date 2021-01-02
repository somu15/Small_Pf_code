#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:50:43 2020

@author: dhulls
"""

import os
import math
import warnings
import time

import numpy as np
import random as rn

from LimitStateFunctions import LimitStateFunctions as LSF
from ML_TF import ML_TF
# from DrawRandom import DrawRandom as DR
from pyDOE import *
from scipy.stats import uniform
from scipy.stats import norm
from BNN import BNN
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import glob
from scipy.interpolate import interp1d



files = glob.glob('/Users/dhulls/projects/Small Pf/VI/ABN_MCS1/*')
for f in files:
    os.remove(f)

# Ninit_GP = 12
# lhd0 = lhs(2, samples=Ninit_GP, criterion='centermaximin')
# lhd = uniform(loc=-3,scale=6).ppf(lhd0)
# lhd0 = lhs(2, samples=4, criterion='centermaximin')
# lhd1 = uniform(loc=-3,scale=6).ppf(lhd0)
# LS1 = LSF()
# trainIn = lhd
# valIn = lhd1
# trainOut = LS1.Scalar_LS1_HF_2D(trainIn)
# valOut = LS1.Scalar_LS1_HF_2D(valIn)
# rv = norm()
# Ninit_GP = 5000
# rv1 = rv.rvs((Ninit_GP,2))

# neu1 = np.array([8,6,4])
# neu2 = np.array([6,4,2])
# count = 0
# num_HF = np.zeros(27)

# for ii in np.arange(0,len(neu1),1):
#     for jj in np.arange(0,len(neu1),1):
#         for kk in np.arange(0,len(neu2),1):
#             files = glob.glob('/Users/dhulls/projects/Small Pf/VI/ABN_MCS1/*')
#             for f in files:
#                 os.remove(f)
#             BNN1 = BNN(obs_ind=trainIn, obs=trainOut, val_ind=valIn, val=valOut)
#             BNN1.Train(neurons1=neu1[ii],neurons2=neu1[jj],neurons3=neu2[kk], layers=3, fileName="ABN_MCS1")
#             predictions = BNN1.Predict(folderName="ABN_MCS1/", InpData=rv1)
            
#             U_new = np.zeros(Ninit_GP)
#             K1 = np.array(predictions).reshape(240,Ninit_GP)
#             for ss in np.arange(0,Ninit_GP,1):
#                 ecdf1 = ECDF((K1[:,ss]-np.mean(K1[:,ss]))/np.std(K1[:,ss]))
#                 f = interp1d(ecdf1.y, ecdf1.x)
#                 U_new[ss] = f(1-0.0233)
#             num_HF[count] = len(np.rot90(np.where(U_new<2)))
#             count = count + 1

    
Ninit_GP = 12
lhd0 = lhs(2, samples=Ninit_GP, criterion='centermaximin')
lhd = uniform(loc=-3,scale=6).ppf(lhd0)
lhd0 = lhs(2, samples=4, criterion='centermaximin')
lhd1 = uniform(loc=-3,scale=6).ppf(lhd0)
LS1 = LSF()

trainIn = lhd
valIn = lhd1
trainOut = LS1.Scalar_LS1_HF_2D(trainIn)
valOut = LS1.Scalar_LS1_HF_2D(valIn)

BNN1 = BNN(obs_ind=trainIn, obs=trainOut, val_ind=valIn, val=valOut)
BNN1.Train(neurons1=4,neurons2=4,neurons3=4, layers=3, fileName="ABN_MCS1")

rv = norm()
Ninit_GP = 90000
rv1 = rv.rvs((Ninit_GP,2))
predictions = BNN1.Predict(folderName="ABN_MCS1/", InpData=rv1)

# out_mean = np.mean(predictions,axis=0).reshape(Ninit_GP)
# out_std = np.std(predictions,axis=0).reshape(Ninit_GP)
# out_actual = LS1.Scalar_LS1_HF_2D(rv1)
# U_func = np.abs(out_mean)/out_std

U_new = np.zeros(Ninit_GP)
K1 = np.array(predictions).reshape(95,Ninit_GP)
for ii in np.arange(0,Ninit_GP,1):
    ecdf1 = ECDF((K1[:,ii]-np.mean(K1[:,ii]))/np.std(K1[:,ii]))
    f = interp1d(ecdf1.y, ecdf1.x)
    U_new[ii] = f(1-0.0233)
    

# ML = ML_TF(obs_ind = trainIn, obs = trainOut)
# amp1, len1, var1 = ML.GP_train(amp_init=1., len_init=1., var_init=1., num_iters = 1000)
# samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = rv1, num_samples=1000)
# U_func_GP = (np.abs(np.mean(np.array(samples1),axis=0)))/np.std(np.array(samples1),axis=0)

req = np.min(U_new)

while req<2:
    files = glob.glob('/Users/dhulls/projects/Small Pf/VI/ABN_MCS1/*')
    for f in files:
        os.remove(f)
    ind = np.where(U_new==np.min(U_new))
    trainIn = np.vstack((trainIn,rv1[ind[0],:]))
    val = LS1.Scalar_LS1_HF_2D(rv1[ind[0],:])
    trainOut = np.concatenate((trainOut,np.array(val).reshape(1)))
    
    BNN1 = BNN(obs_ind=trainIn, obs=trainOut, val_ind=valIn, val=valOut)
    BNN1.Train(neurons1=4,neurons2=4,neurons3=4, layers=3, fileName="ABN_MCS1")
    
    predictions = BNN1.Predict(folderName="ABN_MCS1/", InpData=rv1)
    # out_mean = np.mean(predictions,axis=0).reshape(Ninit_GP)
    # out_std = np.std(predictions,axis=0).reshape(Ninit_GP)
    # out_actual = LS1.Scalar_LS1_HF_2D(rv1)
    U_new = np.zeros(Ninit_GP)
    K1 = np.array(predictions).reshape(95,Ninit_GP)
    for ii in np.arange(0,Ninit_GP,1):
        ecdf1 = ECDF((K1[:,ii]-np.mean(K1[:,ii]))/np.std(K1[:,ii]))
        f = interp1d(ecdf1.y, ecdf1.x)
        U_new[ii] = f(1-0.0233)
    req = np.min(U_new)
