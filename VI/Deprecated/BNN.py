#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:51:30 2020

@author: dhulls
"""

import os
import math
import warnings
import time

import numpy as np
import random as rn
import tensorflow as tf

# from LimitStateFunctions import LimitStateFunctions as LSF
# from ML_TF import ML_TF
# from DrawRandom import DrawRandom as DR
# from pyDOE import *
# from scipy.stats import uniform
# from scipy.stats import norm

from tensorBNN.activationFunctions import Tanh
from tensorBNN.activationFunctions import Softmax
from tensorBNN.layer import DenseLayer
from tensorBNN.network import network
from tensorBNN.predictor import predictor
from tensorBNN.likelihood import GaussianLikelihood
from tensorBNN.metrics import SquaredError, PercentError
# import matplotlib.pyplot as plt

# import time

class BNN:
    
    def __init__(self, obs_ind=None, obs=None, val_ind=None, val=None):
        self.obs_ind = obs_ind
        self.obs = obs
        self.val_ind = val_ind
        self.val = val
        
    def Train(self, neurons1=None, neurons2=None, neurons3=None, layers=None, fileName=None):
        
        inputDims=len(self.obs_ind[0,:])
        outputDims=1

        data=[self.obs_ind, self.obs, self.val_ind, self.val]
        dtype=tf.float32
        
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
        
        width1 = neurons1 # perceptrons per layer
        width2 = neurons2
        width3 = neurons3
        hidden = layers # number of hidden layers
        seed = 0 # random seed
        neuralNet.add(
            DenseLayer( # Dense layer object
                inputDims, # Size of layer input vector
                width1, # Size of layer output vector
                seed=seed, # Random seed
                dtype=dtype)) # Layer datatype
        neuralNet.add(Tanh()) # Tanh activation function
        seed = 1000 # Increment random seed
        neuralNet.add(DenseLayer(width1,
                                     width2,
                                     seed=seed,
                                     dtype=dtype))
            #neuralNet.add(Relu())
        neuralNet.add(Tanh())
        neuralNet.add(DenseLayer(width2,
                                     width3,
                                     seed=seed,
                                     dtype=dtype))
            #neuralNet.add(Relu())
        neuralNet.add(Tanh())
        seed = 1000
        # for n in range(hidden - 1): # Add more hidden layers
        #     neuralNet.add(DenseLayer(width,
        #                              width,
        #                              seed=seed,
        #                              dtype=dtype))
        #     #neuralNet.add(Relu())
        #     neuralNet.add(Tanh())
        #     seed += 1000
        
        neuralNet.add(DenseLayer(width3,
                                 outputDims,
                                 seed=seed,
                                 dtype=dtype))
        
        # width = neurons # perceptrons per layer
        # hidden = layers # number of hidden layers
        # seed = 0 # random seed
        # neuralNet.add(
        #     DenseLayer( # Dense layer object
        #         inputDims, # Size of layer input vector
        #         width, # Size of layer output vector
        #         seed=seed, # Random seed
        #         dtype=dtype)) # Layer datatype
        # neuralNet.add(Tanh()) # Tanh activation function
        # seed += 1000 # Increment random seed
        # for n in range(hidden - 1): # Add more hidden layers
        #     neuralNet.add(DenseLayer(width,
        #                              width,
        #                              seed=seed,
        #                              dtype=dtype))
        #     #neuralNet.add(Relu())
        #     neuralNet.add(Tanh())
        #     seed += 1000
        
        # neuralNet.add(DenseLayer(width,
        #                          outputDims,
        #                          seed=seed,
        #                          dtype=dtype))
        
        neuralNet.setupMCMC(
            0.005, # starting stepsize
            0.0025, # minimum stepsize
            0.01, # maximum stepsize
            40, # number of stepsize options in stepsize adapter
            2, # starting number of leapfrog steps
            2, # minimum number of leapfrog steps
            25, # maximum number of leapfrog steps
            1, # stepsize between leapfrog steps in leapfrog step adapter
            0.0015, # hyper parameter stepsize
            5, # hyper parameter number of leapfrog steps
            20, # number of burnin epochs
            20, # number of cores
            2) # number of averaging steps for param adapters)
        		
        neuralNet.train(
            500, # epochs to train for
            5, # increment between network saves
            likelihood,
            metricList=metricList,
            folderName=fileName, # Name of folder for saved networks
            networksPerFile=5) # Number of networks saved per file
        
    def Predict(self, folderName, InpData=None):
        
        dtype = tf.float32
        pred = predictor(folderName, dtype = dtype )
        
        predictions = pred.predict(InpData,n=1) # ,n=50
        
        return predictions
        
        
        