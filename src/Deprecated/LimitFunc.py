import os
import numpy as np
import scipy.stats as stats
from scipy.stats import norm

class limitfunc:

    def __init__(self, func_name = None):

        if func_name is None:
            raise ValueError('Please provide a limit state function name.')

        self.func_name = func_name

    def evaluate_func(self, params)

        if self.func_name == 'borehole_simcenter'
            # params = [rw, r, Tu, Hu-Hl, Tl, L, Kw]
            value = (2 * np.pi * params[2] * params[3])/(np.log(params[1]/params[0]) * (1 + params[2]/params[4] + (2 * params[5] * params[2])/(np.log(params[1]/params[0]) * np.power(params[0],2) * params[6])))
            return value

        else:
            raise ValueError('Invalid limit state function name.')
