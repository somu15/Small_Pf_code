#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:34:11 2020

@author: dhulls
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:55:13 2020

@author: dhulls
"""

import numpy as np
from scipy.stats import norm

class DrawRandom:
    
    # def __init__(self):
    #     # self.Input_vec = Input_vec
    
    def StandardNormal_Indep(self, N=None):
        out = np.zeros(N)
        rv = norm(loc=0,scale=1)
        for ii in np.arange(0,N,1):
            out[ii] = rv.rvs()
        return out
