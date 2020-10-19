#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:30:39 2020

@author: dhulls
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:11:49 2020

@author: dhulls
"""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

class ML_TF:
    
    def __init__(self, obs_ind=None, obs=None):
        self.obs_ind = obs_ind
        self.obs = obs
        
    def GP_train(self): # Gaussian Process Regression training
        
        def build_gp(amplitude, length_scale, observation_noise_variance):

          kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

          return tfd.GaussianProcess(
              kernel=kernel,
              index_points=self.obs_ind,
              observation_noise_variance=observation_noise_variance)
      
        gp_joint_model = tfd.JointDistributionNamed({
            'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observations': build_gp,
        })
        
        constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

        amplitude_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='amplitude',
            dtype=np.float64)
        
        length_scale_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='length_scale',
            dtype=np.float64)
        
        observation_noise_variance_var = tfp.util.TransformedVariable(
            initial_value=1.,
            bijector=constrain_positive,
            name='observation_noise_variance_var',
            dtype=np.float64)
        
        trainable_variables = [v.trainable_variables[0] for v in 
                               [amplitude_var,
                               length_scale_var,
                               observation_noise_variance_var]]
        
        @tf.function(autograph=False, experimental_compile=False)
        def target_log_prob(amplitude, length_scale, observation_noise_variance):
          return gp_joint_model.log_prob({
              'amplitude': amplitude,
              'length_scale': length_scale,
              'observation_noise_variance': observation_noise_variance,
              'observations': self.obs
          })
      
        num_iters = 1000
        optimizer = tf.optimizers.Adam(learning_rate=.01)
        
        lls_ = np.zeros(num_iters, np.float64)
        for i in range(num_iters):
          with tf.GradientTape() as tape:
            loss = -target_log_prob(amplitude_var, length_scale_var,
                                    observation_noise_variance_var)
          grads = tape.gradient(loss, trainable_variables)
          optimizer.apply_gradients(zip(grads, trainable_variables))
          lls_[i] = loss
          
        print('Trained parameters:')
        print('amplitude: {}'.format(amplitude_var._value().numpy()))
        print('length_scale: {}'.format(length_scale_var._value().numpy()))
        print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))
        
        return amplitude_var, length_scale_var, observation_noise_variance_var
    
    def GP_predict(self, amplitude_var=None, length_scale_var=None, observation_noise_variance_var=None, pred_ind=None, num_samples=None): # Gaussian Process Regression prediction
        
        # Gaussian Process Regression prediction
    
        optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimized_kernel,
            index_points=pred_ind,
            observation_index_points=self.obs_ind,
            observations=self.obs,
            observation_noise_variance=observation_noise_variance_var,
            predictive_noise_variance=0.)
        
        return gprm.sample(num_samples)
        
        
        