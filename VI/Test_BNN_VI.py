#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 10:43:22 2020

@author: dhulls
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
import tensorflow_probability as tfp
from keras import callbacks, optimizers
import tqdm
from scipy.stats import norm
from scipy.stats import uniform
from LimitStateFunctions_new import LimitStateFunctions as LSF
from DrawRandom_new import DrawRandom as DR
from pyDOE import *
from ML_TF_new import ML_TF
from statsmodels.distributions.empirical_distribution import ECDF
import glob
from scipy.interpolate import interp1d
LS1 = LSF()
DR1 = DR()

train_size = 12
noise = 1e-3
np.random.seed(100)
lhd0 = lhs(2, samples=train_size, criterion='centermaximin')
lhd = uniform(loc=-3,scale=3).ppf(lhd0)
X = lhd
y = LS1.Scalar_LS1_HF_2D(X)
y_true = y

prior_std1 = 1.5
prior_std2 = 0.5
prior_std3 = 0.1
prior_p1 = 1.0
prior_p2 = 0.0

batch_size = train_size
num_batches = train_size / batch_size

class DenseVariational(Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=prior_std1,
                 prior_sigma_2=prior_std2,
                 prior_sigma_3=prior_std3,
                 prior_pi_1=prior_p1,
                 prior_pi_2=prior_p2,
                 **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_sigma_3 = prior_sigma_3
        self.prior_pi_1 = prior_pi_1
        self.prior_pi_2 = prior_pi_2
        self.prior_pi_3 = 1 - prior_pi_1 - prior_pi_2
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2 + self.prior_pi_3 * self.prior_sigma_3 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.RandomNormal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.RandomNormal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        comp_3_dist = tfp.distributions.Normal(0.0, self.prior_sigma_3)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w) + self.prior_pi_3 * comp_3_dist.prob(w))

kl_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': prior_std1, 
    'prior_sigma_2': prior_std2,
    'prior_sigma_3': prior_std3, 
    'prior_pi_1': prior_p1,
    'prior_pi_2': prior_p2
}

x_in = Input(shape=(2,))
x = DenseVariational(10, kl_weight, **prior_params, activation='sigmoid')(x_in)
x = DenseVariational(10, kl_weight, **prior_params, activation='sigmoid')(x)
x = DenseVariational(1, kl_weight, **prior_params)(x)

model = Model(x_in, x)

def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    # print(y_pred[0])
    # tf.print(y_pred[0])
    # val = K.std(y_pred-y_obs)
    # tf.print(val)
    dist = tfp.distributions.Normal(loc=y_pred, scale=1e-4) # 0.01 
    return K.sum(-dist.log_prob(y_obs))
    
# opt = tf.keras.optimizers.Adagrad(
#     learning_rate=0.02,
#     initial_accumulator_value=0.1,
#     epsilon=1e-07,
#     name="Adagrad"
# )
# opt1 = tf.keras.optimizers.SGD(
#     learning_rate=0.02, momentum=0.0, nesterov=False, name="SGD")
model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.025), metrics=['mse']) # optimizers.Adam(lr=0.025)
model.fit(X, y, batch_size=batch_size, epochs=3000, verbose=1)

rv = norm()
rv1 = rv.rvs((5000,2))
X_test = rv1 # np.linspace(-3, 3, 1000).reshape(-1, 1)
y_test = LS1.Scalar_LS1_HF_2D(X_test)
y_pred_list = []
for i in tqdm.tqdm(range(200)):
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)
y_preds = np.concatenate(y_pred_list, axis=1)
y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)
U_func = np.abs(y_mean)/y_sigma
req = np.min(U_func)

ML = ML_TF(obs_ind = X, obs = y)
amp1, len1, var1 = ML.GP_train(amp_init=1., len_init=1., var_init=1e-4, num_iters = 1000)
samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = X_test, num_samples=200)
U_func_GP = (np.abs(np.mean(np.array(samples1),axis=0)))/np.std(np.array(samples1),axis=0)

# plt.hist(y_preds[0,:])

# plt.hist(samples1[:,0])

ecdf_BNN = ECDF(y_preds[9,:])
ecdf_GP = ECDF(samples1[:,9])



# while req<2:
#     ind = np.where(U_func==np.min(U_func))
#     X = np.vstack((X,rv1[ind[0],:]))
#     val = LS1.Scalar_LS1_HF_2D(rv1[ind[0],:])
#     y = np.concatenate((y,np.array(val).reshape(1)))
#     x_in = Input(shape=(2,))
#     x = DenseVariational(10, kl_weight, **prior_params, activation='sigmoid')(x_in)
#     x = DenseVariational(10, kl_weight, **prior_params, activation='sigmoid')(x)
#     x = DenseVariational(1, kl_weight, **prior_params)(x)
    
#     model = Model(x_in, x)
#     model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.025), metrics=['mse'])
#     model.fit(X, y, batch_size=batch_size, epochs=1000, verbose=1)
#     y_pred_list = []
#     for i in tqdm.tqdm(range(50)):
#         y_pred = model.predict(X_test)
#         y_pred_list.append(y_pred)
#     y_preds = np.concatenate(y_pred_list, axis=1)
#     y_mean = np.mean(y_preds, axis=1)
#     y_sigma = np.std(y_preds, axis=1)
#     U_func = np.abs(y_mean)/y_sigma
#     req = np.min(U_func)

# plt.scatter(y_test,y_mean)
# plt.xlim([-1,0])
# plt.ylim([-1,0])