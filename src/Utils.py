import os
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import LimitFunc

class Kernel_1D:

    def __init__(self, kernel_name = None):

        if kernel_name is None:
            raise ValueError('Please provide a kernel name.')

        self.kernel_name = kernel_name

    def density(self, rvs, value, bandwidth = 0, bandwidth_rule = None):

        req_value = 0

        if len(rvs) < 2:
            raise ValueError('Random variables array must contain atleast 2 values.')

        if bandwidth <= 0 and bandwidth_rule is None:
            raise ValueError('Either a non-zero bandwidth value or a bandwidth_rule must be provided.')

        if bandwidth > 0:
            band_req = bandwidth
        else:
            if bandwidth_rule.lower() == 'silverman':
                band_req = 1.06 * np.std(rvs) * np.power(len(rvs),-0.2)
            else:
                raise ValueError('Invalid bandwidth rule name.')

        if self.kernel_name.lower() == 'gaussian':

            std_req = np.std(rvs)

            for ind in np.arange(0,len(rvs),1):
                req_value = req_value + norm.pdf((value-rvs[ind])/std_req)
            req_value = req_value/(len(rvs)*band_req)
            return req_value

        else:
            raise ValueError('Invalid kernel name.')

class MCMC:

    def __init__(self, method_name = None):

        if method_name is None:
            raise ValueError('Please provide a method name for MCMC sampling.')

        self.method_name = method_name

    def metropolis(self, mu, sig, ):
