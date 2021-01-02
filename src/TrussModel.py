#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:24:18 2020

@author: dhulls
"""

from anastruct import SystemElements
import numpy as np

class TrussModel:
    
    def HF(self, young1=None, young2=None, area1=None, area2=None, P1=None, P2=None, P3=None, P4=None, P5=None, P6=None):
        
        ss = SystemElements()

        # young1 = 2.1e11
        # area1  = 2e-3
        # young2 = 2.1e11
        # area2  = 1e-3
        ss.add_truss_element(location=[[0, 0], [4,0]], EA=(area1*young1))
        ss.add_truss_element(location=[[4, 0], [8,0]], EA=(area1*young1))
        ss.add_truss_element(location=[[8, 0], [12,0]], EA=(area1*young1))
        ss.add_truss_element(location=[[12, 0], [16,0]], EA=(area1*young1))
        ss.add_truss_element(location=[[16, 0], [20,0]], EA=(area1*young1))
        ss.add_truss_element(location=[[20, 0], [24,0]], EA=(area1*young1))
        ss.add_truss_element(location=[[2, 2], [6,2]], EA=(area1*young1))
        ss.add_truss_element(location=[[6, 2], [10,2]], EA=(area1*young1))
        ss.add_truss_element(location=[[10, 2], [14,2]], EA=(area1*young1))
        ss.add_truss_element(location=[[14, 2], [18,2]], EA=(area1*young1))
        ss.add_truss_element(location=[[18, 2], [22,2]], EA=(area1*young1))
        ss.add_truss_element(location=[[0, 0], [2,2]], EA=(area2*young2))
        ss.add_truss_element(location=[[2,2], [4,0]], EA=(area2*young2))
        ss.add_truss_element(location=[[4,0], [6,2]], EA=(area2*young2))
        ss.add_truss_element(location=[[6,2], [8,0]], EA=(area2*young2))
        ss.add_truss_element(location=[[8,0], [10,2]], EA=(area2*young2))
        ss.add_truss_element(location=[[10,2], [12,0]], EA=(area2*young2))
        ss.add_truss_element(location=[[12,0], [14,2]], EA=(area2*young2))
        ss.add_truss_element(location=[[14,2], [16,0]], EA=(area2*young2))
        ss.add_truss_element(location=[[16,0], [18,2]], EA=(area2*young2))
        ss.add_truss_element(location=[[18,2], [20,0]], EA=(area2*young2))
        ss.add_truss_element(location=[[20,0], [22,2]], EA=(area2*young2))
        ss.add_truss_element(location=[[22,2], [24,0]], EA=(area2*young2))
        
        ss.add_support_hinged(node_id=1)
        ss.add_support_roll(node_id=7, direction='x')
        
        # P1 = -5e4
        # P2 = -5e4
        # P3 = -5e4
        # P4 = -5e4
        # P5 = -5e4
        # P6 = -5e4
        ss.point_load(node_id=8, Fy=P1)
        ss.point_load(node_id=9, Fy=P2)
        ss.point_load(node_id=10, Fy=P3)
        ss.point_load(node_id=11, Fy=P4)
        ss.point_load(node_id=12, Fy=P5)
        ss.point_load(node_id=13, Fy=P6)
        
        ss.solve()
        # ss.show_structure()
        # ss.show_displacement(factor=10)
        K = ss.get_node_results_system(node_id=4)['uy']
        
        return np.array(K)
    
    def LF(self, young1=None, young2=None, area1=None, area2=None, P1=None, P2=None, P3=None, P4=None, P5=None, P6=None):
        
        ss = SystemElements()

        # young1 = 2.1e11
        # area1  = 2e-3
        # young2 = 2.1e11
        # area2  = 1e-3
        ss.add_truss_element(location=[[0, 0], [12,0]], EA=(area1*young1))
        ss.add_truss_element(location=[[12, 0], [24,0]], EA=(area1*young1))
        ss.add_truss_element(location=[[6, 2], [18,2]], EA=(area1*young1))
        ss.add_truss_element(location=[[0, 0], [6,2]], EA=(area2*young2))
        ss.add_truss_element(location=[[6,2], [12,0]], EA=(area2*young2))
        ss.add_truss_element(location=[[12,0], [18,2]], EA=(area2*young2))
        ss.add_truss_element(location=[[18,2], [24,0]], EA=(area2*young2))
        
        ss.add_support_hinged(node_id=1)
        ss.add_support_roll(node_id=3, direction='x')
        
        # P1 = -5e4
        # P2 = -5e4
        # P3 = -5e4
        # P4 = -5e4
        # P5 = -5e4
        # P6 = -5e4
        ss.point_load(node_id=4, Fy=np.sum([P1,P2,P3]))
        ss.point_load(node_id=5, Fy=np.sum([P4,P5,P6]))
        
        ss.solve()
        # ss.show_structure()
        # ss.show_displacement(factor=10)
        K = ss.get_node_results_system(node_id=4)['uy']
        
        return np.array(K)


