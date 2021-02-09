#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:55:13 2020

@author: dhulls
"""

import numpy as np
from scipy.stats import norm
from TrussModel import TrussModel as TM
from MaterialModel import MaterialModel as Mat
from FluidModel import FluidModel as Fluid
from HolesModel import HolesModel as Holes

class LimitStateFunctions:
    
    # def __init__(self):
    #     # self.Input_vec = Input_vec
    
    def Scalar_LS1(self, Input_vec=None): # sin(sqrt(x1^2 + x2^2))
        return np.sin(np.sqrt(Input_vec[:,0]**2+Input_vec[:,1]**2))
    
    def Scalar_LS2(self, Input_vec=None): # sqrt(x1^2 + x2^2 + x3^2)
        return np.sqrt((Input_vec[:,0]**2+Input_vec[:,1]**2+Input_vec[:,2]**2))
    
    def Scalar_LS3(self, Input_vec=None): # sin(3 * pi * x)
        return np.sin(3 * np.pi * Input_vec[:,0])
    
    def Scalar_LS4(self, Input_vec=None): # sum(xi^5) - 5 * sum(xi) + 12
        return (Input_vec[:,0]**5+Input_vec[:,1]**5+Input_vec[:,2]**5) - 5 * (Input_vec[:,0]+Input_vec[:,1]+Input_vec[:,2]) + 12
    
    def Scalar_LS1_HF(self, Input_vec=None):
        return (Input_vec[:,0]**2)
    
    def Scalar_LS1_LF(self, Input_vec=None):
        return (Input_vec[:,0])
    
    def Scalar_LS2_HF(self, Input_vec=None):
        k = norm(loc=0,scale=0.25)
        return (Input_vec[:,0]**2 + k.rvs())
    
    def Scalar_LS2_LF(self, Input_vec=None):
        return (Input_vec[:,0]**2)
    
    def Scalar_LS1_HF_2D(self, Input_vec=None):
        # Echard et al (2011) Example 1
        k = 6
        y1 = 3 + 0.1 * (Input_vec[:,0]-Input_vec[:,1])**2 - (Input_vec[:,0]+Input_vec[:,1]) / np.sqrt(2)
        y2 = 3 + 0.1 * (Input_vec[:,0]-Input_vec[:,1])**2 + (Input_vec[:,0]+Input_vec[:,1]) / np.sqrt(2)
        y3 = (Input_vec[:,0]-Input_vec[:,1]) + k / np.sqrt(2)
        y4 = (Input_vec[:,1]-Input_vec[:,0]) + k / np.sqrt(2)
        return np.min([y1,y2,y3,y4],axis=0)
    
    # def Scalar_LS1_LF_2D(self, Input_vec=None):
    #     # Echard et al (2011) Example 1 modified with noise
    #     k = 6
    #     k1 = norm(loc=0,scale=0.25)
    #     tmp =  + k1.rvs()
    #     y1 = 7 + 0.25 * (Input_vec[:,0]-Input_vec[:,1]) - (Input_vec[:,0]+Input_vec[:,1]) / np.sqrt(2) + tmp
    #     y2 = 7 + 0.25 * (Input_vec[:,0]-Input_vec[:,1]) + (Input_vec[:,0]+Input_vec[:,1]) / np.sqrt(2) + tmp
    #     y3 = (Input_vec[:,0]-Input_vec[:,1]) + k / np.sqrt(2) + tmp
    #     y4 = (Input_vec[:,1]-Input_vec[:,0]) + k / np.sqrt(2) + tmp
    #     return (np.min([y1,y2,y3,y4],axis=0)) # 
    
    # def Scalar_LS1_HF_2D(self, Input_vec=None):
    #     # Echard et al (2011) Example 2 [Rastrigin function]
    #     return (10-(Input_vec[:,0]**2-5*np.cos(2*np.pi*Input_vec[:,0]))-(Input_vec[:,1]**2-5*np.cos(2*np.pi*Input_vec[:,1])))
    
    def Scalar_Borehole_HF_nD(self, Input_vec=None):
        
        # Input_vec = [rw, r, Tu, Hu, Tl, Hl, L, Kw]
        #             [0,  1, 2,  3,  4,  5,  6,  7]
        
        numer = 2*np.pi*Input_vec[:,2]*(Input_vec[:,3]-Input_vec[:,5])
        denom = np.log(np.exp(Input_vec[:,1])/Input_vec[:,0]) * (1 + (2*Input_vec[:,6]*Input_vec[:,2])/(np.log(np.exp(Input_vec[:,1])/Input_vec[:,0]) * Input_vec[:,0]**2 * Input_vec[:,7]) + (Input_vec[:,2]) / (Input_vec[:,4]))
        return (numer/denom)
    
    def Truss_HF(self, Input_vec=None):
        
        # Input_vec = [E1, E2, A1, A2, P1, P2, P3, P4, P5, P6]
        #             [0,  1,  2,  3,  4,  5,  6,  7,  8,  9]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        TM1 = TM()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(TM1.HF(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5],Input_vec[ii,6],Input_vec[ii,7],Input_vec[ii,8],Input_vec[ii,9]))
        
        return out1
    
    def Truss_LF(self, Input_vec=None):
        
        # Input_vec = [E1, E2, A1, A2, P1, P2, P3, P4, P5, P6]
        #             [0,  1,  2,  3,  4,  5,  6,  7,  8,  9]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        TM1 = TM()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(TM1.LF(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5],Input_vec[ii,6],Input_vec[ii,7],Input_vec[ii,8],Input_vec[ii,9]))
        
        return out1
    
    def Material_HF(self, Input_vec=None):
        
        # Input_vec = [Ex, Ez, vxy, vxz, Gxz, ux, uy, uz]
        #             [0,  1,   2,   3,   4,   5,  6,  7]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Mat1 = Mat()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Mat1.HF(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5],Input_vec[ii,6],Input_vec[ii,7]))
        
        return out1
    
    def Material_HF1(self, Input_vec=None):
        
        # Input_vec = [Ex, Ez, vxy, vxz, Gxz, ux, uy, uz]
        #             [0,  1,   2,   3,   4,   5,  6,  7]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Mat1 = Mat()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Mat1.HF1(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5],Input_vec[ii,6],Input_vec[ii,7]))
        
        return out1
    
    def Material_LF1(self, Input_vec=None):
        
        # Input_vec = [E, v, ux, uy, uz]
        #             [0, 1,  2,  3,  4]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Mat1 = Mat()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Mat1.LF1(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4]))
        
        return out1
    
    def Fluid_NS(self, Input_vec=None):
        
        # Input_vec = [viscosity, density, u_Bottom, u_Top, u_Left, u_Right]
        #             [0,               1,        2,     3,      4,      5]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Fluid1 = Fluid()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Fluid1.Navier_Stokes(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5]))
        
        return out1
    
    def Fluid_S(self, Input_vec=None):
        
        # Input_vec = [viscosity, density, u_Bottom, u_Top, u_Left, u_Right]
        #             [0,               1,        2,     3,      4,      5]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Fluid1 = Fluid()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Fluid1.Stokes(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5]))
        
        return out1
    
   
    def Fluid_NS1(self, Input_vec=None):
        
        # Input_vec = [viscosity, density, u_Bottom, u_Top, u_Left, u_Right]
        #             [0,               1,        2,     3,      4,      5]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Fluid1 = Fluid()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Fluid1.Navier_Stokes1(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5]))
        
        return out1

    def Fluid_S1(self, Input_vec=None):
        
        # Input_vec = [viscosity, density, u_Bottom, u_Top, u_Left, u_Right]
        #             [0,               1,        2,     3,      4,      5]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Fluid1 = Fluid()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Fluid1.Stokes1(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5]))
        
        return out1
    
    def Holes_HF(self, Input_vec=None):
        
        # Input_vec = [E, v, ux]
        #             [0, 1,  2]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Holes1 = Holes()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Holes1.HF(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2]))
        
        return out1
    
    def Holes_HF1(self, Input_vec=None):
        
        # Input_vec = [E, v, ux]
        #             [0, 1,  2]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Holes1 = Holes()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Holes1.HF1(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2]))
        
        return out1
    
    def Holes_LF1(self, Input_vec=None):
        
        # Input_vec = [E, v, ux]
        #             [0, 1,  2]
        
        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        Holes1 = Holes()
        
        for ii in np.arange(0,siz1,1):
            out1[ii] = np.abs(Holes1.LF1(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2]))
        
        return out1

