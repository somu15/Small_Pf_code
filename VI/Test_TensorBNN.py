#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:52:08 2020

@author: dhulls
"""

import os
import math
import warnings
import time

import numpy as np
import random as rn
import tensorflow as tf

from LimitStateFunctions import LimitStateFunctions as LSF
from ML_TF import ML_TF
from DrawRandom import DrawRandom as DR
from pyDOE import *
from scipy.stats import uniform
from scipy.stats import norm

from tensorBNN.activationFunctions import Tanh
from tensorBNN.layer import DenseLayer
from tensorBNN.network import network
from tensorBNN.predictor import predictor
from tensorBNN.likelihood import GaussianLikelihood
from tensorBNN.metrics import SquaredError, PercentError
import matplotlib.pyplot as plt

import time

startTime = time.time()

# This supresses many deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)
rn.seed(12345)
tf.random.set_seed(3)


# def main():

Ninit_GP = 12
lhd0 = lhs(2, samples=Ninit_GP, criterion='centermaximin')
lhd = uniform(loc=-3,scale=6).ppf(lhd0)
lhd0 = lhs(2, samples=2, criterion='centermaximin')
lhd1 = uniform(loc=-3,scale=6).ppf(lhd0)
LS1 = LSF()

# trainIn=np.linspace(-2,2,num=31)
trainIn = lhd
# valIn=np.linspace(-2+2/30,2.0-2/30,num=3)
valIn = lhd1
# trainOut = np.sin(trainIn*math.pi*2)*trainIn-np.cos(trainIn*math.pi)
trainOut = LS1.Scalar_LS1_HF_2D(trainIn)
# valOut = np.sin(valIn*math.pi*2)*valIn-np.cos(valIn*math.pi)
valOut = LS1.Scalar_LS1_HF_2D(valIn)


data=[trainIn, trainOut, valIn, valOut]

dtype=tf.float32

inputDims=2
outputDims=1

normInfo=(0,1) # mean, sd

likelihood=GaussianLikelihood(sd=0.1)
metricList=[SquaredError(mean=normInfo[0], sd=normInfo[1]), PercentError(mean=normInfo[0], sd=normInfo[1])]

neuralNet = network(
            dtype, # network datatype
            inputDims, # dimension of input vector
            data[0], # training input data
            data[1], # training output data
            data[2], # validation input data
            data[3]) # validation output data

width = 10 # perceptrons per layer
hidden = 3 # number of hidden layers
seed = 0 # random seed
neuralNet.add(
    DenseLayer( # Dense layer object
        inputDims, # Size of layer input vector
        width, # Size of layer output vector
        seed=seed, # Random seed
        dtype=dtype)) # Layer datatype
neuralNet.add(Tanh()) # Tanh activation function
seed += 1000 # Increment random seed
for n in range(hidden - 1): # Add more hidden layers
    neuralNet.add(DenseLayer(width,
                             width,
                             seed=seed,
                             dtype=dtype))
    #neuralNet.add(Relu())
    neuralNet.add(Tanh())
    seed += 1000

neuralNet.add(DenseLayer(width,
                         outputDims,
                         seed=seed,
                         dtype=dtype))

neuralNet.setupMCMC(
    0.005, # starting stepsize
    0.0025, # minimum stepsize
    0.01, # maximum stepsize
    40, # number of stepsize options in stepsize adapter
    2, # starting number of leapfrog steps
    2, # minimum number of leapfrog steps
    50, # maximum number of leapfrog steps
    1, # stepsize between leapfrog steps in leapfrog step adapter
    0.0015, # hyper parameter stepsize
    5, # hyper parameter number of leapfrog steps
    20, # number of burnin epochs
    20, # number of cores
    2) # number of averaging steps for param adapters)
		
neuralNet.train(
    3000, # epochs to train for
    1, # increment between network saves
    likelihood,
    metricList=metricList,
    folderName="Test_TBNN", # Name of folder for saved networks
    networksPerFile=50) # Number of networks saved per file
        
    # print("Time elapsed:", time.time() - startTime)
    
pred = predictor("Test_TBNN/", dtype = dtype )

rv = norm()
Ninit_GP = 90000
rv1 = rv.rvs((Ninit_GP,2))

pred_input = rv1
predictions = pred.predict(pred_input,n=1)

out_mean = np.mean(predictions,axis=0).reshape(Ninit_GP)
out_std = np.std(predictions,axis=0).reshape(Ninit_GP)
out_actual = LS1.Scalar_LS1_HF_2D(pred_input)
U_func = np.abs(out_mean)/out_std

# in_data = np.linspace(-2,2,num=400)
# plt.plot(in_data,out_mean,in_data,out_mean-3*out_std,in_data,out_mean+3*out_std)
# plt.scatter(trainIn,trainOut)

# samps = np.zeros(2950)
# for ii in np.arange(0,2950):
#     samps[ii] = predictions[ii][0,200]
