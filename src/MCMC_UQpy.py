#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:09:40 2020

@author: dhulls
"""

def rosenbrock_no_params(x):
      return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/20)
 

# x = MH(dimension=2, pdf_target=rosenbrock_no_params, nburn=500, jump=50, nsamples=500, nchains=1)
# print(x.samples.shape)
# plt.plot(x.samples[:,0], x.samples[:,1], 'o', alpha=0.5)
# plt.show()

# proposal = [Normal(), Normal()]
# proposal_is_symmetric = [False, False]

# x = MMH(dimension=2, nburn=1, jump=50, nsamples=1, pdf_target=rosenbrock_no_params,
#         proposal=proposal, proposal_is_symmetric=proposal_is_symmetric, nchains=1)

# fig, ax = plt.subplots()
# ax.plot(x.samples[:, 0], x.samples[:, 1], linestyle='none', marker='.')