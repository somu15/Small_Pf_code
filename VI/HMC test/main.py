#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:56:26 2020

@author: dhulls
"""

# import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from bnn import (
    get_random_initial_state,
    target_log_prob_fn_factory,
    tracer_factory, build_net,
)
from hmc import predict_from_chain, run_hmc, pre_train_nn
from map import get_best_map_state, get_map_trace, get_nodes_per_layer
import matplotlib.pyplot as plt
# %%
# About the data: https://kaggle.com/c/boston-housing
train, test = tf.keras.datasets.boston_housing.load_data()
X_train, y_train, X_test, y_test = [arr.astype("float32") for arr in [*train, *test]]
# %%
weight_prior = tfp.distributions.Normal(0, 0.2)
bias_prior = tfp.distributions.Normal(0, 0.2)
# %%
log_prob_tracers = (
    tracer_factory(X_train, y_train),
    tracer_factory(X_test, y_test),
)
n_features = X_train.shape[-1]
random_initial_state = get_random_initial_state(
    weight_prior, bias_prior, get_nodes_per_layer(n_features)
)

# nodes_per_layer = get_nodes_per_layer(n_features)
# w_prior = weight_prior
# b_prior = bias_prior
# overdisp = 1.0
# init_state = []
# for n1, n2 in zip(nodes_per_layer, nodes_per_layer[1:]):
#     w_shape, b_shape = [n1, n2], n2
#     # Use overdispersion > 1 for better R-hat statistics.
#     w = w_prior.sample(w_shape) * overdisp
#     b = b_prior.sample(b_shape) * overdisp
#     init_state.append([tf.Variable(w), tf.Variable(b)])
# random_initial_state = init_state


# @tf.function(experimental_compile=True)
# def sample_chain(*args, **kwargs):
#     """@tf.function JIT-compiles a static graph for tfp.mcmc.sample_chain.
#     This significantly improves performance, especially when enabling XLA.
#     https://tensorflow.org/xla#explicit_compilation_with_tffunction
#     https://github.com/tensorflow/probability/issues/728#issuecomment-573704750
#     """
#     return tfp.mcmc.sample_chain(*args, **kwargs)
# step_size = 1e-3
# kernel = tfp.mcmc.NoUTurnSampler(bnn_log_prob_fn, step_size=step_size)
# adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
#     kernel, num_adaptation_steps=500
# )
# chain, trace, final_kernel_results = sample_chain(
#     kernel=adaptive_kernel,
#     num_results=2000,
#     current_state=random_initial_state,
#     return_final_kernel_results=True,
# )


trace, log_probs = get_map_trace(
    target_log_prob_fn_factory(weight_prior, bias_prior, X_train, y_train),
    random_initial_state,
    n_iter=3000,
    callbacks=log_prob_tracers,
)
best_map_params = get_best_map_state(trace, log_probs)
# %%
map_nn = build_net(best_map_params)
map_y_pred, map_y_var = map_nn(X_test, training=False)
# %%
bnn_log_prob_fn = target_log_prob_fn_factory(weight_prior, bias_prior, X_train, y_train)
# burnin, samples, trace, final_kernel_results = run_hmc(bnn_log_prob_fn, current_state=best_map_params)
# hmc_y_pred, hmc_y_var = predict_from_chain(samples, X_test)

weights,model=pre_train_nn(X_train, y_train, get_nodes_per_layer(n_features), epochs=1000)

# @tf.function(experimental_compile=True)
def sample_chain(*args, **kwargs):
    """@tf.function JIT-compiles a static graph for tfp.mcmc.sample_chain.
    This significantly improves performance, especially when enabling XLA.
    https://tensorflow.org/xla#explicit_compilation_with_tffunction
    https://github.com/tensorflow/probability/issues/728#issuecomment-573704750
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)
step_size = 1e-3
kernel = tfp.mcmc.NoUTurnSampler(bnn_log_prob_fn, step_size=step_size)
adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    kernel, num_adaptation_steps=500
)
chain, trace, final_kernel_results = sample_chain(
    kernel=kernel,
    num_results=2000,
    current_state=weights,
    return_final_kernel_results=True, 
)

# hmc_y_pred, hmc_y_var = predict_from_chain(chain, X_test)

restructured_chain = [
            [tensor[i] for tensor in chain] for i in range(len(chain[0]))
        ]

def predict(params):
    post_model = build_net(params)
    y_pred, y_var = post_model(X_test, training=False)
    return y_pred, y_var

preds = [predict(chunks(params, 2)) for params in restructured_chain]
k1,k2=tf.unstack(preds,axis=1)
plt.plot(k1[:,0])


