#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:41:07 2020

@author: dhulls
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:53:29 2020

@author: dhulls
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from LimitStateFunctions import LimitStateFunctions as LSF
from ML_TF import ML_TF

## Testing GP for 1D inputs

def generate_1d_data(num_training_points, observation_noise_variance):
  
  index_points_ = np.random.uniform(-2., 2., (num_training_points, 1))
  index_points_ = index_points_.astype(np.float64)
  LS1 = LSF()
  observations_ = (LS1.Scalar_LS3(Input_vec=index_points_) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))
  return index_points_, observations_

NUM_TRAINING_POINTS = 150
obs_ind_, obs_ = generate_1d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=.25)

predictive_index_points_ = np.linspace(-2., 2., 75)[:,None]

ML = ML_TF(obs_ind = obs_ind_, obs = obs_)

amp1, len1, var1 = ML.GP_train()

num_s = 50
samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = predictive_index_points_, num_samples=num_s)

fig = plt.figure()
for i in range(num_s):
  plt.plot(predictive_index_points_, samples1[i, :], c='r', alpha=.1,
            label='Posterior Sample' if i == 0 else None)
plt.plot(predictive_index_points_, np.mean(np.array(samples1),axis=0), 
          label='Posterior Mean',c='r', alpha=1, linewidth=0.5)
plt.scatter(obs_ind_, obs_, s=20, label='Observations')
plt.legend(loc='lower left')
plt.show()


## Testing GP for 2D inputs

def generate_1d_data(num_training_points, observation_noise_variance):
  
  index_points_ = np.random.uniform(-2., 2., (num_training_points, 2))
  index_points_ = index_points_.astype(np.float64)
  LS1 = LSF()
  observations_ = (LS1.Scalar_LS1(Input_vec=index_points_) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))
  return index_points_, observations_

NUM_TRAINING_POINTS = 150
obs_ind_, obs_ = generate_1d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=.01)

xv = np.linspace(-2., 2., 75)
yv = np.linspace(-2., 2., 75)
predictive_index_points_ = np.zeros((len(xv)*len(yv),2))
count = 0
for i in np.arange(0,len(xv),1):
    for j in np.arange(0,len(yv),1):
        predictive_index_points_[count,0] = xv[i]
        predictive_index_points_[count,1] = yv[j]
        count = count + 1

ML = ML_TF(obs_ind = obs_ind_, obs = obs_)

amp1, len1, var1 = ML.GP_train()

num_s = 50
samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = predictive_index_points_, num_samples=num_s)

X1,Y1 = np.meshgrid(xv,yv)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X1,Y1, np.mean(np.array(samples1),axis=0).reshape(75,75), cmap=cm.seismic,alpha=0.1,
                      linewidth=0, antialiased=False,
          label='Posterior Sample' if i == 0 else None)
ax.set_zlim(0.11, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.scatter(obs_ind_[:, 0], obs_ind_[:, 1], obs_, zdir='z', s=20)
plt.legend(loc='upper right')
plt.show()


# # ### Testing 1D GP from TensorFlow

# # ## Defining the required function

# # def sinusoid(x):
# #   return x[..., 0]**2 #np.sin(3 * np.pi * x[..., 0])

# # def generate_1d_data(num_training_points, observation_noise_variance):
# #   """Generate noisy sinusoidal observations at a random set of points.

# #   Returns:
# #      observation_index_points, observations
# #   """
# #   index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
# #   index_points_ = index_points_.astype(np.float64)
# #   # y = f(x) + noise
# #   observations_ = (sinusoid(index_points_) +
# #                    np.random.normal(loc=0,
# #                                     scale=np.sqrt(observation_noise_variance),
# #                                     size=(num_training_points)))
# #   return index_points_, observations_

# # NUM_TRAINING_POINTS = 100
# # obs_ind_, obs_ = generate_1d_data(
# #     num_training_points=NUM_TRAINING_POINTS,
# #     observation_noise_variance=.1)

# # # plt.scatter(obs_ind_,obs_)

# # ## Priors over the hyperparameters

# # def build_gp(amplitude, length_scale, observation_noise_variance):
# #   """Defines the conditional dist. of GP outputs, given kernel parameters."""

# #   kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

# #   # Create the GP prior distribution, which we will use to train the model
# #   # parameters.
# #   return tfd.GaussianProcess(
# #       kernel=kernel,
# #       index_points=obs_ind_,
# #       observation_noise_variance=observation_noise_variance)

# # gp_joint_model = tfd.JointDistributionNamed({
# #     'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
# #     'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
# #     'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
# #     'observations': build_gp,
# # })

# # ## Optimization of the hyperparameters

# # constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

# # amplitude_var = tfp.util.TransformedVariable(
# #     initial_value=1.,
# #     bijector=constrain_positive,
# #     name='amplitude',
# #     dtype=np.float64)

# # length_scale_var = tfp.util.TransformedVariable(
# #     initial_value=1.,
# #     bijector=constrain_positive,
# #     name='length_scale',
# #     dtype=np.float64)

# # observation_noise_variance_var = tfp.util.TransformedVariable(
# #     initial_value=1.,
# #     bijector=constrain_positive,
# #     name='observation_noise_variance_var',
# #     dtype=np.float64)

# # trainable_variables = [v.trainable_variables[0] for v in 
# #                        [amplitude_var,
# #                        length_scale_var,
# #                        observation_noise_variance_var]]

# # @tf.function(autograph=False, experimental_compile=False)
# # def target_log_prob(amplitude, length_scale, observation_noise_variance):
# #   return gp_joint_model.log_prob({
# #       'amplitude': amplitude,
# #       'length_scale': length_scale,
# #       'observation_noise_variance': observation_noise_variance,
# #       'observations': obs_
# #   })

# # num_iters = 1000
# # optimizer = tf.optimizers.Adam(learning_rate=.01)

# # # Store the likelihood values during training, so we can plot the progress

# # lls_ = np.zeros(num_iters, np.float64)
# # for i in range(num_iters):
# #   with tf.GradientTape() as tape:
# #     loss = -target_log_prob(amplitude_var, length_scale_var,
# #                             observation_noise_variance_var)
# #   grads = tape.gradient(loss, trainable_variables)
# #   optimizer.apply_gradients(zip(grads, trainable_variables))
# #   lls_[i] = loss

# # print('Trained parameters:')
# # print('amplitude: {}'.format(amplitude_var._value().numpy()))
# # print('length_scale: {}'.format(length_scale_var._value().numpy()))
# # print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

# # plt.figure(figsize=(12, 4))
# # plt.plot(lls_)
# # plt.xlabel("Training iteration")
# # plt.ylabel("Log marginal likelihood")
# # plt.show()

# # ## Predict

# # predictive_index_points_ = np.linspace(-1.2, 1.2, 200, dtype=np.float64)
# # # Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
# # predictive_index_points_ = predictive_index_points_[..., np.newaxis]

# # optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
# # gprm = tfd.GaussianProcessRegressionModel(
# #     kernel=optimized_kernel,
# #     index_points=predictive_index_points_,
# #     observation_index_points=obs_ind_,
# #     observations=obs_,
# #     observation_noise_variance=observation_noise_variance_var,
# #     predictive_noise_variance=0.)

# # # Create op to draw  50 independent samples, each of which is a *joint* draw
# # # from the posterior at the predictive_index_points_. Since we have 200 input
# # # locations as defined above, this posterior distribution over corresponding
# # # function values is a 200-dimensional multivariate Gaussian distribution!
# # num_samples = 50
# # samples = gprm.sample(num_samples)

# # plt.figure(figsize=(12, 4))
# # plt.plot(predictive_index_points_, sinusoid(predictive_index_points_),
# #          label='True fn')
# # plt.scatter(obs_ind_[:, 0], obs_,
# #             label='Observations')
# # for i in range(num_samples):
# #   plt.plot(predictive_index_points_, samples[i, :], c='r', alpha=.1,
# #            label='Posterior Sample' if i == 0 else None)
# # leg = plt.legend(loc='upper right')
# # for lh in leg.legendHandles: 
# #     lh.set_alpha(1)
# # plt.xlabel(r"Index points ($\mathbb{R}^1$)")
# # plt.ylabel("Observation space")
# # plt.show()


# ### Testing nD GP from TensorFlow

# ## Defining the required function

# def nD_func(x):
#   return np.sin(np.sqrt(x[:,0]**2+x[:,1]**2)) # np.sqrt((x[:,0]**2+x[:,1]**2))

# def generate_1d_data(num_training_points, observation_noise_variance):
#   """Generate noisy sinusoidal observations at a random set of points.

#   Returns:
#      observation_index_points, observations
#   """
#   index_points_ = np.random.uniform(-2., 2., (num_training_points, 2))
#   index_points_ = index_points_.astype(np.float64)
#   # y = f(x) + noise
#   observations_ = (nD_func(index_points_) +
#                    np.random.normal(loc=0,
#                                     scale=np.sqrt(observation_noise_variance),
#                                     size=(num_training_points)))
#   return index_points_, observations_

# NUM_TRAINING_POINTS = 150
# obs_ind_, obs_ = generate_1d_data(
#     num_training_points=NUM_TRAINING_POINTS,
#     observation_noise_variance=.01)

# # plt.scatter(obs_ind_,obs_)

# ## Priors over the hyperparameters

# def build_gp(amplitude, length_scale, observation_noise_variance):
#   """Defines the conditional dist. of GP outputs, given kernel parameters."""

#   kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

#   # Create the GP prior distribution, which we will use to train the model
#   # parameters.
#   return tfd.GaussianProcess(
#       kernel=kernel,
#       index_points=obs_ind_,
#       observation_noise_variance=observation_noise_variance)

# gp_joint_model = tfd.JointDistributionNamed({
#     'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
#     'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
#     'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
#     'observations': build_gp,
# })

# ## Optimization of the hyperparameters

# constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

# amplitude_var = tfp.util.TransformedVariable(
#     initial_value=1.,
#     bijector=constrain_positive,
#     name='amplitude',
#     dtype=np.float64)

# length_scale_var = tfp.util.TransformedVariable(
#     initial_value=1.,
#     bijector=constrain_positive,
#     name='length_scale',
#     dtype=np.float64)

# observation_noise_variance_var = tfp.util.TransformedVariable(
#     initial_value=1.,
#     bijector=constrain_positive,
#     name='observation_noise_variance_var',
#     dtype=np.float64)

# trainable_variables = [v.trainable_variables[0] for v in 
#                        [amplitude_var,
#                        length_scale_var,
#                        observation_noise_variance_var]]

# @tf.function(autograph=False, experimental_compile=False)
# def target_log_prob(amplitude, length_scale, observation_noise_variance):
#   return gp_joint_model.log_prob({
#       'amplitude': amplitude,
#       'length_scale': length_scale,
#       'observation_noise_variance': observation_noise_variance,
#       'observations': obs_
#   })

# num_iters = 1000
# optimizer = tf.optimizers.Adam(learning_rate=.01)

# # Store the likelihood values during training, so we can plot the progress

# lls_ = np.zeros(num_iters, np.float64)
# for i in range(num_iters):
#   with tf.GradientTape() as tape:
#     loss = -target_log_prob(amplitude_var, length_scale_var,
#                             observation_noise_variance_var)
#   grads = tape.gradient(loss, trainable_variables)
#   optimizer.apply_gradients(zip(grads, trainable_variables))
#   lls_[i] = loss

# print('Trained parameters:')
# print('amplitude: {}'.format(amplitude_var._value().numpy()))
# print('length_scale: {}'.format(length_scale_var._value().numpy()))
# print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

# plt.figure(figsize=(12, 4))
# plt.plot(lls_)
# plt.xlabel("Training iteration")
# plt.ylabel("Log marginal likelihood")
# plt.show()

# ## Predict


# # xv, yv = np.meshgrid(np.linspace(-1.2, 1.2, 50), np.linspace(-1.2, 1.2, 50))

# xv = np.linspace(-2., 2., 75)
# yv = np.linspace(-2., 2., 75)
# predictive_index_points_ = np.zeros((len(xv)*len(yv),2))
# count = 0
# for i in np.arange(0,len(xv),1):
#     for j in np.arange(0,len(yv),1):
#         predictive_index_points_[count,0] = xv[i]
#         predictive_index_points_[count,1] = yv[j]
#         count = count + 1
        
# # Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
# # predictive_index_points_ = predictive_index_points_[..., np.newaxis]

# optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
# gprm = tfd.GaussianProcessRegressionModel(
#     kernel=optimized_kernel,
#     index_points=predictive_index_points_,
#     observation_index_points=obs_ind_,
#     observations=obs_,
#     observation_noise_variance=observation_noise_variance_var,
#     predictive_noise_variance=0.)

# # Create op to draw  50 independent samples, each of which is a *joint* draw
# # from the posterior at the predictive_index_points_. Since we have 200 input
# # locations as defined above, this posterior distribution over corresponding
# # function values is a 200-dimensional multivariate Gaussian distribution!
# num_samples = 50
# samples = gprm.sample(num_samples)

# X1,Y1 = np.meshgrid(xv,yv)
# Z1 = nD_func(predictive_index_points_).reshape(75,75)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(obs_ind_[:, 0], obs_ind_[:, 1], obs_, zdir='z', s=20)
# ax.plot_surface(X1,Y1, np.mean(np.array(samples),axis=0).reshape(75,75), cmap=cm.seismic,alpha=0.1,
#                       linewidth=0, antialiased=False,
#           label='Posterior Sample' if i == 0 else None)
# # for i in range(num_samples):
# #   ax.plot_surface(X1,Y1, np.array(samples[i,:]).reshape(75,75), cmap=cm.coolwarm, alpha=0.1,
# #                         linewidth=0, antialiased=False,
# #             label='Posterior Sample' if i == 0 else None)
# ax.plot_surface(X1, Y1, Z1, cmap=cm.coolwarm, alpha=1.0,
#                         linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# plt.show()



# index1 = 10
# mr = np.mean(samples[:,index1])
# sr = np.std(samples[:,index1])
# Xr = np.arange(mr-1.5,mr+1.5,0.05)
# dens_pred = norm.pdf(Xr,loc=mr,scale=sr)
# plt.plot(Xr,dens_pred)
# plt.plot([obs_[index1],obs_[index1]],[0.,np.max(dens_pred)])
# plt.show()


# # plt.figure(figsize=(12, 4))
# # plt.plot(predictive_index_points_, nD_func(predictive_index_points_),
# #          label='True fn')
# # plt.scatter(obs_ind_[:, 0], obs_,
# #             label='Observations')
# # for i in range(num_samples):
# #   plt.plot(predictive_index_points_, samples[i, :], c='r', alpha=.1,
# #            label='Posterior Sample' if i == 0 else None)
# # leg = plt.legend(loc='upper right')
# # for lh in leg.legendHandles: 
# #     lh.set_alpha(1)
# # plt.xlabel(r"Index points ($\mathbb{R}^1$)")
# # plt.ylabel("Observation space")
# # plt.show()