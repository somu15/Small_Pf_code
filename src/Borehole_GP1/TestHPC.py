#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:00:45 2022

@author: dhulls
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:54:29 2022

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
lhd = DR1.BoreholeLHS(Ninit_GP) #  uniform(loc=-3.5,scale=7.0).ppf(lhd0) #
inp_LFtrain = lhd
y_HF_LFtrain = LS1.Scalar_Borehole_HF_nD(inp_LFtrain)
ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,P,Ndim), obs = Norm3(y_HF_LFtrain,y_HF_LFtrain))
amp0, len0 = ML0.GP_train(amp_init=1.0, len_init=1.0, num_iters = 1000)

print(amp0)
print(len0)