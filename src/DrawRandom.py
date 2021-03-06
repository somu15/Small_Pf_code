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
from scipy.stats import uniform
from scipy.stats import gumbel_r
from pyDOE import *

class DrawRandom:
    
    # def __init__(self):
    #     # self.Input_vec = Input_vec
    
    def StandardNormal_Indep(self, N=None):
        out = np.zeros(N)
        rv = norm(loc=0,scale=1)
        for ii in np.arange(0,N,1):
            out[ii] = rv.rvs()
        return out
    
    def BoreholeRandom(self):
        out = np.zeros(8)
        rv1 = norm(loc=np.log(7.71),scale=1.0056)
        rv2 = uniform()
        out[0] = 0.05 + 0.1 * rv2.rvs()
        out[1] = (rv1.rvs())
        out[2] = 63070 + 52530 * rv2.rvs()
        out[3] = 990 + 120 * rv2.rvs()
        out[4] = 63.1 + 52.9 * rv2.rvs()
        out[5] = 700 + 120 * rv2.rvs()
        out[6] = 1120 + 560 * rv2.rvs()
        out[7] = 9855 + 2190 * rv2.rvs()
        return out
    
    
    def BoreholePDF(self, rv_req=None, index=None):
        if index == 0:
            rv = uniform(loc=0.05,scale=0.1)
            out = rv.pdf((rv_req))
        elif index == 1:
            rv = norm(loc=np.log(7.71),scale=1.0056)
            out = rv.pdf((rv_req))
        elif index == 2:
            rv = uniform(loc=63070,scale=52530)
            out = rv.pdf((rv_req))
        elif index == 3:
            rv = uniform(loc = 990, scale=120)
            out = rv.pdf((rv_req))
        elif index == 4:
            rv = uniform(loc=63.1,scale=52.9)
            out = rv.pdf((rv_req))
        elif index == 5:
            rv = uniform(loc=700,scale=120)
            out = rv.pdf((rv_req))
        elif index == 6:
            rv = uniform(loc=1120,scale=560)
            out = rv.pdf((rv_req))
        else:
            rv = uniform(loc=9855,scale=2190)
            out = rv.pdf((rv_req))
        return out

    def BoreholeLHS(self, Nsamps=None):
        out = np.zeros((Nsamps,8))
        lhd0 = lhs(8, samples=Nsamps, criterion='centermaximin')
        out[:,0] = uniform(loc=0.05,scale=0.1).ppf(lhd0[:,0])
        out[:,1] = uniform(loc=-2.0,scale=9.0).ppf(lhd0[:,1])  # norm(loc=np.log(7.71),scale=1.0056).ppf(lhd0[:,1])
        out[:,2] = uniform(loc=63070,scale=52530).ppf(lhd0[:,2])
        out[:,3] = uniform(loc = 990, scale=120).ppf(lhd0[:,3])
        out[:,4] = uniform(loc=63.1,scale=52.9).ppf(lhd0[:,4])
        out[:,5] = uniform(loc=700,scale=120).ppf(lhd0[:,5])
        out[:,6] = uniform(loc=1120,scale=560).ppf(lhd0[:,6])
        out[:,7] = uniform(loc=9855,scale=2190).ppf(lhd0[:,7])
        return out
    
    def TrussRandom(self):
        out = np.zeros(10)
        rv1 = norm(loc=26.0653,scale=0.099) # E1 and E2
        rv2 = norm(loc=-6.2195,scale=0.09975) # A1
        rv3 = norm(loc=-6.9127,scale=0.099503) # A2
        rv4 = gumbel_r(loc=5e4,scale=(7.5e3/1.28254)) # P1, P2, P3, P4, P5, and P6 
        out[0] = np.exp(rv1.rvs())
        out[1] = np.exp(rv1.rvs())
        out[2] = np.exp(rv2.rvs())
        out[3] = np.exp(rv3.rvs())
        out[4] = rv4.rvs()
        out[5] = rv4.rvs()
        out[6] = rv4.rvs()
        out[7] = rv4.rvs()
        out[8] = rv4.rvs()
        out[9] = rv4.rvs()
        return out
    
    def TrussPDF(self, rv_req=None, index=None):
        if index == 0 or index==1:
            rv = norm(loc=26.0653,scale=0.099) # E1 and E2
            out = rv.pdf(np.log(rv_req))
        elif index == 2:
            rv = norm(loc=-6.2195,scale=0.09975) # A1
            out = rv.pdf(np.log(rv_req))
        elif index==3:
            rv = norm(loc=-6.9127,scale=0.099503) # A2
            out = rv.pdf(np.log(rv_req))
        else:
            rv = gumbel_r(loc=5e4,scale=(7.5e3/1.28254)) # P1, P2, P3, P4, P5, and P6
            out = rv.pdf((rv_req))
        return out
    
    def TrussLHS(self, Nsamps=None):
        out = np.zeros((Nsamps,10))
        lhd0 = lhs(10, samples=Nsamps, criterion='centermaximin')
        out[:,0] = uniform(loc=1.365e11,scale=1.47e11).ppf(lhd0[:,0]) # uniform(loc=25.7188,scale=0.693).ppf(lhd0[:,0])
        out[:,1] = uniform(loc=1.365e11,scale=1.47e11).ppf(lhd0[:,1]) # uniform(loc=25.7188,scale=0.693).ppf(lhd0[:,1])
        out[:,2] = uniform(loc=1.3e-3,scale=1.4e-3).ppf(lhd0[:,2]) # uniform(loc=-6.56862,scale=0.69824).ppf(lhd0[:,2])
        out[:,3] = uniform(loc=6.5e-4,scale=7e-4).ppf(lhd0[:,3]) # uniform(loc=-7.26095, scale=0.6965).ppf(lhd0[:,3])
        out[:,4] = uniform(loc=23750,scale=52500).ppf(lhd0[:,4]) # uniform(loc=23750,scale=52500).ppf(lhd0[:,4])
        out[:,5] = uniform(loc=23750,scale=52500).ppf(lhd0[:,5])
        out[:,6] = uniform(loc=23750,scale=52500).ppf(lhd0[:,6])
        out[:,7] = uniform(loc=23750,scale=52500).ppf(lhd0[:,7])
        out[:,8] = uniform(loc=23750,scale=52500).ppf(lhd0[:,8])
        out[:,9] = uniform(loc=23750,scale=52500).ppf(lhd0[:,9])
        return out
    
    def MaterialRandom(self):
        out = np.zeros(8)
        rv1 = norm(loc=np.log(200),scale=0.1) # Ex
        rv2 = norm(loc=np.log(300),scale=0.1) # Ez
        rv3 = norm(loc=np.log(0.25),scale=0.1) # vxy
        rv4 = norm(loc=np.log(0.3),scale=0.1) # vxz
        rv5 = norm(loc=np.log(135),scale=0.1) # Gxz
        rv6 = norm(loc=np.log(0.0025),scale=0.1) # ux, uy, uz
        out[0] = np.exp(rv1.rvs())
        out[1] = np.exp(rv2.rvs())
        out[2] = np.exp(rv3.rvs())
        out[3] = np.exp(rv4.rvs())
        out[4] = np.exp(rv5.rvs())
        out[5] = np.exp(rv6.rvs())
        out[6] = np.exp(rv6.rvs())
        out[7] = np.exp(rv6.rvs())
        out1 = np.zeros(5)
        out1[0] = out[0]
        out1[1] = out[2]
        out1[2] = out[5]
        out1[3] = out[6]
        out1[4] = out[7]
        return out, out1
    
    def MaterialLHS(self, Nsamps=None):
        out = np.zeros((Nsamps,8))
        lhd0 = lhs(8, samples=Nsamps, criterion='centermaximin')
        out[:,0] = uniform(loc=140.937,scale=142.876).ppf(lhd0[:,0]) # Ex
        out[:,1] = uniform(loc=211.406,scale=214.314).ppf(lhd0[:,1]) # Ez
        out[:,2] = uniform(loc=0.176,scale=0.178).ppf(lhd0[:,2]) # vxy
        out[:,3] = uniform(loc=0.211,scale=0.214).ppf(lhd0[:,3]) # vxz
        out[:,4] = uniform(loc=95.132,scale=96.442).ppf(lhd0[:,4]) # Gxz
        out[:,5] = uniform(loc=0.00176,scale=0.001787).ppf(lhd0[:,5]) # ux
        out[:,6] = uniform(loc=0.00176,scale=0.001787).ppf(lhd0[:,6]) # uy
        out[:,7] = uniform(loc=0.00176,scale=0.001787).ppf(lhd0[:,7]) # uz
        out1 = np.zeros((Nsamps,5))
        out1[:,0] = out[:,0]
        out1[:,1] = out[:,2]
        out1[:,2] = out[:,5]
        out1[:,3] = out[:,6]
        out1[:,4] = out[:,7]
        return out, out1
    
    def MaterialPDF(self, rv_req=None, index=None, LF=None):
        
        if LF==0:
            
            if index == 0:
                rv = norm(loc=np.log(200),scale=0.1) # Ex
                out = rv.pdf(np.log(rv_req))
            elif index == 1:
                rv = norm(loc=np.log(300),scale=0.1) # Ez
                out = rv.pdf(np.log(rv_req))
            elif index == 2:
                rv = norm(loc=np.log(0.25),scale=0.1) # vxy
                out = rv.pdf(np.log(rv_req))
            elif index == 3:
                rv = norm(loc=np.log(0.3),scale=0.1) # vxz
                out = rv.pdf(np.log(rv_req))
            elif index == 4:
                rv = norm(loc=np.log(135),scale=0.1) # Gxz
                out = rv.pdf(np.log(rv_req))
            else:
                rv = norm(loc=np.log(0.0025),scale=0.1) # ux, uy, uz
                out = rv.pdf(np.log(rv_req))
                
        else:
            
            if index == 0:
                rv = norm(loc=np.log(200),scale=0.1) # Ex
                out = rv.pdf(np.log(rv_req))
            elif index == 1:
                rv = norm(loc=np.log(0.25),scale=0.1) # vxy
                out = rv.pdf(np.log(rv_req))
            else:
                rv = norm(loc=np.log(0.0025),scale=0.1) # ux, uy, uz
                out = rv.pdf(np.log(rv_req))
                
        return out
            
        