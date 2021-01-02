#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:15:33 2020

@author: dhulls
"""

# Imports
import numpy as np
np.random.seed(100)
from tensorflow import random
random.set_seed(100)

import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
os.chdir('/Users/som/Dropbox/Complex_systems_RNN/DL_tutorial')
from Kij import Kij
from scipy.stats import beta

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
print(tf.__version__)
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
# from matplotlib import rc

# Import dataset

# dataset_path = '/Users/som/Dropbox/Complex_systems_RNN/Data/DL_test_data_Eq_Hazus_125.csv'
dataset_path = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for improved disf match/Eq_10IM.csv'
# column_names = ['Time','P1','P2','IM','Rec']

# dataset = pd.read_csv(dataset_path, names=column_names,
#                       na_values = "?", comment='\t',
#                       sep=" ", skipinitialspace=True)

dataset = pd.read_csv(dataset_path)
dataset.pop("IM")
dataset.pop("Time")
# dataset["IM"] = np.log(dataset["IM"])
# dataset.pop("P1")
# dataset.pop("P2")
a1 = 1.0
b1 = 1.0
loc1 = 0
sca1 = 1

def transform(Rec):
    # Rec = beta.ppf(Rec,a1,b1,loc1,sca1)
    # Fin = np.zeros((len(Rec),1))
    # for ii in np.arange(0,len(Rec),1):
    #     if ((1-Rec[ii])<0.04):
    #         Fin[ii] = np.power((1-Rec[ii]),1/4)
    #     else:
    #         Fin[ii] = 1-Rec[ii]
    # return Fin
    return np.power((1-Rec),1/4)
    # return (1/(1+np.exp(-(1-Rec))))

def invtransform(x):
    # Fin = np.zeros(len(x))
    # for ii in np.arange(0,len(x),1):
    #     if ((1-np.power(x[ii],4))<0.04):
    #         Fin[ii] = (1-np.power(x[ii],4))
    #     else:
    #         Fin[ii] = 1-x[ii]
    # return Fin # (1-np.power(x,4))
    return (1-np.power(x,4))
    # return (1+np.log(1/(x)-1))
    #beta.cdf(x,a1,b1,loc1,sca1)

dataset['Rec'] = transform(dataset['Rec'])

dataset['P1'] = transform(dataset['P1'])

dataset['P2'] = transform(dataset['P2'])

dataset.tail()

# Split the data into train and test

train_dataset = dataset.sample(frac=1.0,random_state=100)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data

# sns.pairplot(train_dataset[["Rec", "P1", "P2", "Time"]], diag_kind="kde")
# sns.pairplot(train_dataset[["Rec", "IM"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("Rec")
train_stats = train_stats.transpose()
train_stats

# Split features from labels

train_labels = train_dataset.pop('Rec')
test_labels = test_dataset.pop('Rec')

# Normalize the data

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Build the model ,kernel_regularizer='l2'

def build_model():
          model = keras.Sequential([
            layers.Dense(10, activation='softmax', input_shape=[len(train_dataset.keys())],bias_initializer='zeros'),
            layers.Dense(10, activation='softmax',bias_initializer='zeros'),
            layers.Dense(1,bias_initializer='zeros')
          ])
        
          # optimizer = tf.keras.optimizers.RMSprop(0.001)
          optimizer = tf.keras.optimizers.RMSprop(0.001)
        
          model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model

model = build_model()

# Inspect the model

model.summary()

# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# example_result

# Train the model

EPOCHS = 3000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.0, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()],shuffle = False)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


## Verify model for multiple recovery curves

dataset_path1 = '/Users/som/Dropbox/Complex_systems_RNN/Data/New data for sequence/DL_verify_EQ_0_7.csv'
data1 = pd.read_csv(dataset_path1)
# data1["IM"] = np.log(data1["IM"])
Time1 = data1.pop("Time")
# data1.pop("P1")
# data1.pop("P2")
data1['Rec'] = transform(data1['Rec'])
data1['P1'] = transform(data1['P1'])
data1['P2'] = transform(data1['P2'])
data1_labels = data1.pop('Rec')
normed_data1 = norm(data1)
data1_pred = model.predict(normed_data1).flatten()