#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:04:32 2020

@author: dhulls
"""

from __future__ import print_function
from __future__ import absolute_import
from argparse import ArgumentParser
import numpy as nm

import sys
sys.path.append('.')

from sfepy.base.base import IndexedStruct, Struct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess.viewer import Viewer
from sfepy.postprocess.probes_vtk import ProbeFromFile, Probe
from sfepy.mechanics.matcoefs import stiffness_from_lame
from sfepy.mechanics.matcoefs import stiffness_for_transIso
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
import numpy as np

class MaterialModel:
    
    def HF(self, Ex, Ez, vxy, vxz, Gxz, ux, uy, uz):
        
        def shift_u_fun(ts, coors, bc=None, problem=None, shift=0.0):
            """
            Define a displacement depending on the y coordinate.
            """
            val = shift * coors[:,1]**2
        
            return val
        
        helps = {
            'show' : 'show the results figure',
        }
        
        def post_process(out, problem, state, extend=False):
            """
            Calculate and output strain and stress for given displacements.
            """
        
            ev = problem.evaluate
            
            stress = ev('ev_cauchy_stress.%d.Omega(m.D, u)' % (2), mode='el_avg',
                        copy_materials=False, verbose=False)
            out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                          data=stress, dofs=None)
        
            return out
        
        from sfepy import data_dir
        
        parser = ArgumentParser()
        parser.add_argument('--version', action='version', version='%(prog)s')
        parser.add_argument('-s', '--show',
                            action="store_true", dest='show',
                            default=False, help=helps['show'])
        options = parser.parse_args()
        
        mesh = Mesh.from_file(data_dir + '/meshes/3d/cylinder1.inp')
        domain = FEDomain('domain', mesh)
        
        omega = domain.create_region('Omega', 'all')
        Bottom = domain.create_region('Bottom', 'vertices in (z < -0.049)', 'facet')
        Top = domain.create_region('Top', 'vertices in (z > 0.049)', 'facet')
        
        field = Field.from_args('fu', nm.float64, 'vector', omega, approx_order=1)
        
        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')
        
        # m = Material('m', D=stiffness_from_youngpoisson(dim=3, young=2, poisson=0.3))
        
        m = Material('m', D=stiffness_for_transIso(dim=3, Ex=Ex, Ez=Ez, vxy=vxy, vxz=vxz, Gxz=Gxz))
        
        integral = Integral('i', order=2)
        
        t1 = Term.new('dw_lin_elastic(m.D, v, u)',
                      integral, omega, m=m, v=v, u=u)
        eq = Equation('balance', t1)
        eqs = Equations([eq])
        
        Fixed = EssentialBC('Fixed', Bottom, {'u.all' : 0.0})
        Displaced  = EssentialBC('Displaced', Top, {'u.0' : ux, 'u.1' : uy, 'u.2' : uz})
        
        ls = ScipyDirect({})
        
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status)
        
        pb = Problem('elasticity', equations=eqs)
        
        pb.save_regions_as_groups('regions')
        
        pb.set_bcs(ebcs=Conditions([Fixed, Displaced]))
        
        pb.set_solver(nls)
        
        status = IndexedStruct()
        
        #####
        
        state = pb.solve(save_results=True, post_process_hook=post_process)   # status=status, 
        
        out = state.create_output_dict()
        out = post_process(out, pb, state, extend=True)
        pb.save_state('postprocess.vtk', out=out)
        
        prb = Probe(out,pb.domain.mesh)
        
        def Compute_Max_VonMises(probe,num_points,interval):
            radii = np.arange(interval,0.05,interval)
            lens = np.arange(interval,0.05,interval)
            max_stress = 0
            for ii in np.arange(0,len(lens),1):
                for jj in np.arange(0,len(radii),1):
                    probe.add_circle_probe('Test', np.array([0,0,lens[ii]]), np.array([0,0,1]), radii[jj], num_points)
                    pars,vals = prb.__call__('Test','cauchy_stress')
                    req = np.max(0.5*((vals[:,0]-vals[:,1])**2+(vals[:,2]-vals[:,1])**2+(vals[:,0]-vals[:,2])**2+6*(vals[:,3]**2+vals[:,4]**2+vals[:,5]**2)))
                    if max_stress < req:
                        max_stress = req
            return max_stress
                    
        stress_von = Compute_Max_VonMises(prb,500,0.0015)
        
        return stress_von
    
    def LF(self, E1, v1, ux, uy, uz):
        
        def shift_u_fun(ts, coors, bc=None, problem=None, shift=0.0):
            """
            Define a displacement depending on the y coordinate.
            """
            val = shift * coors[:,1]**2
        
            return val
        
        helps = {
            'show' : 'show the results figure',
        }
        
        def post_process(out, problem, state, extend=False):
            """
            Calculate and output strain and stress for given displacements.
            """
        
            ev = problem.evaluate
            
            stress = ev('ev_cauchy_stress.%d.Omega(m.D, u)' % (2), mode='el_avg',
                        copy_materials=False, verbose=False)
            out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                          data=stress, dofs=None)
        
            return out
        
        from sfepy import data_dir
        
        parser = ArgumentParser()
        parser.add_argument('--version', action='version', version='%(prog)s')
        parser.add_argument('-s', '--show',
                            action="store_true", dest='show',
                            default=False, help=helps['show'])
        options = parser.parse_args()
        
        mesh = Mesh.from_file(data_dir + '/meshes/3d/cylinder.inp')
        domain = FEDomain('domain', mesh)
        
        omega = domain.create_region('Omega', 'all')
        Bottom = domain.create_region('Bottom', 'vertices in (z < -0.049)', 'facet')
        Top = domain.create_region('Top', 'vertices in (z > 0.049)', 'facet')
        
        field = Field.from_args('fu', nm.float64, 'vector', omega, approx_order=1)
        
        u = FieldVariable('u', 'unknown', field)
        v = FieldVariable('v', 'test', field, primary_var_name='u')
        
        mu1 = E1/(2.0*(1.0 + v1))
        lam1 = E1*v1/((1.0 + v1)*(1.0 - 2.0*v1))
        
        m = Material('m', D=stiffness_from_lame(dim=3, mu=mu1, lam=lam1))
        
        # m = Material('m', D=stiffness_from_youngpoisson(dim=3, young=E, poisson=v))
        
        # m = Material('m', D=stiffness_for_transIso(dim=3, Ex=Ex, Ez=Ez, vxy=vxy, vxz=vxz, Gxz=Gxz))
        
        integral = Integral('i', order=2)
        
        t1 = Term.new('dw_lin_elastic(m.D, v, u)',
                      integral, omega, m=m, v=v, u=u)
        eq = Equation('balance', t1)
        eqs = Equations([eq])
        
        Fixed = EssentialBC('Fixed', Bottom, {'u.all' : 0.0})
        Displaced  = EssentialBC('Displaced', Top, {'u.0' : ux, 'u.1' : uy, 'u.2' : uz})
        
        ls = ScipyDirect({})
        
        nls_status = IndexedStruct()
        nls = Newton({}, lin_solver=ls, status=nls_status)
        
        pb = Problem('elasticity', equations=eqs)
        
        pb.save_regions_as_groups('regions')
        
        pb.set_bcs(ebcs=Conditions([Fixed, Displaced]))
        
        pb.set_solver(nls)
        
        status = IndexedStruct()
        
        #####
        
        state = pb.solve(save_results=True, post_process_hook=post_process)   # status=status, 
        
        out = state.create_output_dict()
        out = post_process(out, pb, state, extend=True)
        pb.save_state('postprocess.vtk', out=out)
        
        prb = Probe(out,pb.domain.mesh)
        
        def Compute_Max_VonMises(probe,num_points,interval):
            radii = np.arange(interval,0.05,interval)
            lens = np.arange(interval,0.05,interval)
            max_stress = 0
            for ii in np.arange(0,len(lens),1):
                for jj in np.arange(0,len(radii),1):
                    probe.add_circle_probe('Test', np.array([0,0,lens[ii]]), np.array([0,0,1]), radii[jj], num_points)
                    pars,vals = prb.__call__('Test','cauchy_stress')
                    req = np.max(0.5*((vals[:,0]-vals[:,1])**2+(vals[:,2]-vals[:,1])**2+(vals[:,0]-vals[:,2])**2+6*(vals[:,3]**2+vals[:,4]**2+vals[:,5]**2)))
                    if max_stress < req:
                        max_stress = req
            return max_stress
                    
        stress_von = Compute_Max_VonMises(prb,500,0.0015)
        
        return stress_von
