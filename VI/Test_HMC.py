#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:31:14 2020

@author: dhulls
"""

import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions
normals_2d = [
    tfd.MultivariateNormalDiag([2, -2]),
    tfd.MultivariateNormalDiag([-2, 2]),
]
bimodal = tfd.Mixture(tfd.Categorical([0.5, 0.5]), normals_2d)
@tf.function(experimental_compile=True)
def sample_chain(*args, **kwargs):
    """@tf.function JIT-compiles a static graph for tfp.mcmc.sample_chain.
    This significantly improves performance, especially when enabling XLA.
    https://tensorflow.org/xla#explicit_compilation_with_tffunction
    https://github.com/tensorflow/probability/issues/728#issuecomment-573704750
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)
step_size = 1e-3
kernel = tfp.mcmc.NoUTurnSampler(bimodal.log_prob, step_size=step_size)
adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    kernel, num_adaptation_steps=500
)
chain, trace, final_kernel_results = sample_chain(
    kernel=adaptive_kernel,
    num_results=2000,
    current_state=tf.constant([0.1, 0]),
    return_final_kernel_results=True,
)
xr = np.linspace(-6, 6, 13)
domain = np.stack(np.meshgrid(xr, xr), -1).reshape(-1, 2)
density_plot = go.Surface(
    x=xr, y=xr, z=bimodal.prob(domain).numpy().reshape(len(xr), -1)
)
samples_plot = go.Scatter3d(x=chain[:, 0], y=chain[:, 1], z=bimodal.prob(chain))
fig = go.Figure(data=[density_plot, samples_plot])
fig.update_layout(height=700, title_text=f"step size: {step_size}")