#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:39:23 2021

@author: dhulls
"""

from __future__ import print_function
from __future__ import absolute_import
from argparse import ArgumentParser
# import numpy as nm

import os
import sys
import csv
sys.path.append('.')

# from sfepy.base.base import IndexedStruct, Struct
# from sfepy.discrete import (FieldVariable, Material, Integral, Function,
#                             Equation, Equations, Problem)
# from sfepy.discrete.fem import Mesh, FEDomain, Field
# from sfepy.terms import Term
# from sfepy.discrete.conditions import Conditions, EssentialBC, InitialCondition
# from sfepy.solvers.ls import ScipyDirect, ScipySuperLU, ScipyIterative
# from sfepy.solvers.nls import Newton, ScipyBroyden, PETScNonlinearSolver
# from sfepy.solvers.oseen import Oseen
# from sfepy.solvers.oseen import StabilizationFunction
# from sfepy.postprocess.viewer import Viewer
# from sfepy.postprocess.probes_vtk import ProbeFromFile, Probe
import numpy as np

# helps = {
#     'show' : 'show the results figure',
# }

# from sfepy import data_dir

# parser = ArgumentParser()
# parser.add_argument('--version', action='version', version='%(prog)s')
# parser.add_argument('-s', '--show',
#                     action="store_true", dest='show',
#                     default=False, help=helps['show'])
# options = parser.parse_args()

# def Compute_Max_Pressure(probe,num_points,interval):
#     lens = np.arange(-0.5,0.5,interval)
#     max_press = 0
#     for ii in np.arange(0,len(lens),1):
#         p0 = [lens[ii],  -0.5, 0]
#         p1 = [lens[ii], 0.5, 0]
#         probe.add_line_probe('Test', p0, p1, num_points)
#         pars,vals = probe.__call__('Test','p')
#         req = np.max(np.abs(vals))
#         if max_press < req:
#             max_press = req
#     return max_press

class FluidModel:
    
    def Navier_Stokes(self, viscosity, density, u_Bottom, u_Top, u_Left, u_Right):
        
        # mesh = Mesh.from_file(data_dir + '/meshes/3d/fluid_lid.inp')
        # domain = FEDomain('domain', mesh)
        
        # omega = domain.create_region('Omega', 'all')
        # field_1 = Field.from_args(name='3_velocity', dtype=nm.float64, shape=2, region=omega, approx_order=2)
        # field_2 = Field.from_args(name='pressure', dtype=nm.float64, shape=1, region=omega, approx_order=1)
        
        # region_0 = domain.create_region(name='Right', select='vertices in (x > 0.499)', kind='facet')
        # region_1 = domain.create_region(name='Left', select='vertices in (x < -0.499)', kind='facet')
        # region_2 = domain.create_region(name='Top', select='vertices in (y > 0.499)', kind='facet')
        # region_3 = domain.create_region(name='Bottom', select='vertices in (y < -0.499)', kind='facet')
        # region_4 = domain.create_region(name='Point', select='vertices in (x < -0.49) & (y < -0.49)', kind='vertex')
        
        # ebc_1 = EssentialBC(name='Right', region=region_0, dofs={'u.0' : 0.0,'u.1' : u_Right}) # 2.15
        # ebc_2 = EssentialBC(name='Left', region=region_1, dofs={'u.0' : 0.0,'u.1' : u_Left}) # -2.25
        # ebc_3 = EssentialBC(name='Bottom', region=region_3, dofs={'u.0' : u_Bottom,'u.1' : 0.0}) # -2.0
        # ebc_4 = EssentialBC(name='Top', region=region_2, dofs={'u.0' : u_Top,'u.1' : 0.0}) # 2.5
        # ebc_5 = EssentialBC(name='Point', region=region_4, dofs={'p.0' : 1e-9})
        
        # viscosity = Material(name='viscosity', value=viscosity) # 1e-3
        # density = Material(name='density', value=density) # 30.0
        
        # variable_1 = FieldVariable('u', 'unknown', field_1)
        # variable_2 = FieldVariable(name='v', kind='test', field=field_1, primary_var_name='u')
        # variable_3 = FieldVariable(name='p', kind='unknown', field=field_2)
        # variable_4 = FieldVariable(name='q', kind='test', field=field_2, primary_var_name='p')
        
        # integral_1 = Integral('i1', order=2) # 2 
        # integral_2 = Integral('i2', order=2) # 3
        
        # t1 = Term.new(name='dw_div_grad(viscosity.value, v, u)',
        #               integral=integral_2, region=omega, viscosity=viscosity, v=variable_2, u=variable_1)
        # t2 = Term.new(name='dw_convect(v, u)',
        #               integral=integral_2, region=omega, v=variable_2, u=variable_1)
        # t3 = Term.new(name='dw_stokes(density.value, v, p)',
        #               integral=integral_1, density=density, region=omega, v=variable_2, p=variable_3)
        # t4 = Term.new(name='dw_stokes(u, q)',
        #               integral=integral_1, region=omega, u=variable_1, q=variable_4)
        # eq1 = Equation('balance', t1+t2-t3)
        # eq2 = Equation('incompressibility', t4)
        # eqs = Equations([eq1,eq2])
        
        # ls = ScipySuperLU({})
        # nls_status = IndexedStruct()
        # nls = Newton({'i_max' : 2000, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'or', 'macheps' : 1e-16, 'lin_red' : 1e-4, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-2}, lin_solver=ls, status=nls_status)
        # pb = Problem('Navier-Stokes', equations=eqs)
        # pb.set_bcs(ebcs=Conditions([ebc_1, ebc_2, ebc_3, ebc_4, ebc_5]))
        # pb.set_solver(nls)
        # status = IndexedStruct()
        # state = pb.solve(status=status, save_results=True)
        
        # out = state.create_output_dict()
        # pb.save_state('Navier_Stokes.vtk', out=out)
        
        # prb = Probe(out,pb.domain.mesh)
        
        # prb.add_points_probe('center', [[0.0,0.0,0.0]])
        # pars,vals = prb.__call__('center','u')
        
        # resultant = np.sqrt(vals[:,0]**2+vals[:,1]**2)
        
        # return resultant
        
        visc = viscosity
        dens = density
        utop = u_Top
        ubot = u_Bottom
        uright = u_Right
        uleft = u_Left
        
        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS.i', 'r') 
        Lines = file1.readlines()
        Lines[113] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
        Lines[80] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
        Lines[86] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
        Lines[92] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
        Lines[98] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"
        
        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS.i', 'w') 
        file1.writelines(Lines) 
        file1.close() 
        
        os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
        os.system('mpiexec -n 2 /Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i NS.i')
        
        path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            Samp1 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                    Samp1.append(float(row[2]))
                count = count + 1
        
        resultant_NS = np.sqrt(Samp0[1]**2+Samp1[1]**2)
        return resultant_NS
    
    def Navier_Stokes1(self, viscosity, density, u_Bottom, u_Top, u_Left, u_Right):
        
        # mesh = Mesh.from_file(data_dir + '/meshes/3d/fluid_lid.inp')
        # domain = FEDomain('domain', mesh)
        
        # omega = domain.create_region('Omega', 'all')
        # field_1 = Field.from_args(name='3_velocity', dtype=nm.float64, shape=2, region=omega, approx_order=2)
        # field_2 = Field.from_args(name='pressure', dtype=nm.float64, shape=1, region=omega, approx_order=1)
        
        # region_0 = domain.create_region(name='Right', select='vertices in (x > 0.499)', kind='facet')
        # region_1 = domain.create_region(name='Left', select='vertices in (x < -0.499)', kind='facet')
        # region_2 = domain.create_region(name='Top', select='vertices in (y > 0.499)', kind='facet')
        # region_3 = domain.create_region(name='Bottom', select='vertices in (y < -0.499)', kind='facet')
        # region_4 = domain.create_region(name='Point', select='vertices in (x < -0.49) & (y < -0.49)', kind='vertex')
        
        # ebc_1 = EssentialBC(name='Right', region=region_0, dofs={'u.0' : 0.0,'u.1' : u_Right}) # 2.15
        # ebc_2 = EssentialBC(name='Left', region=region_1, dofs={'u.0' : 0.0,'u.1' : u_Left}) # -2.25
        # ebc_3 = EssentialBC(name='Bottom', region=region_3, dofs={'u.0' : u_Bottom,'u.1' : 0.0}) # -2.0
        # ebc_4 = EssentialBC(name='Top', region=region_2, dofs={'u.0' : u_Top,'u.1' : 0.0}) # 2.5
        # ebc_5 = EssentialBC(name='Point', region=region_4, dofs={'p.0' : 1e-9})
        
        # viscosity = Material(name='viscosity', value=viscosity) # 1e-3
        # density = Material(name='density', value=density) # 30.0
        
        # variable_1 = FieldVariable('u', 'unknown', field_1)
        # variable_2 = FieldVariable(name='v', kind='test', field=field_1, primary_var_name='u')
        # variable_3 = FieldVariable(name='p', kind='unknown', field=field_2)
        # variable_4 = FieldVariable(name='q', kind='test', field=field_2, primary_var_name='p')
        
        # integral_1 = Integral('i1', order=2) # 2 
        # integral_2 = Integral('i2', order=2) # 3
        
        # t1 = Term.new(name='dw_div_grad(viscosity.value, v, u)',
        #               integral=integral_2, region=omega, viscosity=viscosity, v=variable_2, u=variable_1)
        # t2 = Term.new(name='dw_convect(v, u)',
        #               integral=integral_2, region=omega, v=variable_2, u=variable_1)
        # t3 = Term.new(name='dw_stokes(density.value, v, p)',
        #               integral=integral_1, density=density, region=omega, v=variable_2, p=variable_3)
        # t4 = Term.new(name='dw_stokes(u, q)',
        #               integral=integral_1, region=omega, u=variable_1, q=variable_4)
        # eq1 = Equation('balance', t1+t2-t3)
        # eq2 = Equation('incompressibility', t4)
        # eqs = Equations([eq1,eq2])
        
        # ls = ScipySuperLU({})
        # nls_status = IndexedStruct()
        # nls = Newton({'i_max' : 2000, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'or', 'macheps' : 1e-16, 'lin_red' : 1e-4, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-2}, lin_solver=ls, status=nls_status)
        # pb = Problem('Navier-Stokes', equations=eqs)
        # pb.set_bcs(ebcs=Conditions([ebc_1, ebc_2, ebc_3, ebc_4, ebc_5]))
        # pb.set_solver(nls)
        # status = IndexedStruct()
        # state = pb.solve(status=status, save_results=True)
        
        # out = state.create_output_dict()
        # pb.save_state('Navier_Stokes.vtk', out=out)
        
        # prb = Probe(out,pb.domain.mesh)
        
        # prb.add_points_probe('center', [[0.0,0.0,0.0]])
        # pars,vals = prb.__call__('center','u')
        
        # resultant = np.sqrt(vals[:,0]**2+vals[:,1]**2)
        
        # return resultant
        
        visc = viscosity
        dens = density
        utop = u_Top
        ubot = u_Bottom
        uright = u_Right
        uleft = u_Left
        
        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS1.i', 'r') 
        Lines = file1.readlines()
        Lines[113] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
        Lines[80] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
        Lines[86] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
        Lines[92] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
        Lines[98] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"
        
        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS1.i', 'w') 
        file1.writelines(Lines) 
        file1.close() 
        
        os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
        os.system('mpiexec -n 2 /Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i NS1.i')
        
        path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS1.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            Samp1 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                    Samp1.append(float(row[2]))
                count = count + 1
        
        resultant_NS = np.sqrt(Samp0[1]**2+Samp1[1]**2)
        return resultant_NS
    
    def Stokes(self, viscosity, density, u_Bottom, u_Top, u_Left, u_Right):
        
        # mesh = Mesh.from_file(data_dir + '/meshes/3d/fluid_lid.inp')
        # domain = FEDomain('domain', mesh)
        
        # omega = domain.create_region('Omega', 'all')
        # field_1 = Field.from_args(name='3_velocity', dtype=nm.float64, shape=2, region=omega, approx_order=2)
        # field_2 = Field.from_args(name='pressure', dtype=nm.float64, shape=1, region=omega, approx_order=1)
        
        # region_0 = domain.create_region(name='Right', select='vertices in (x > 0.499)', kind='facet')
        # region_1 = domain.create_region(name='Left', select='vertices in (x < -0.499)', kind='facet')
        # region_2 = domain.create_region(name='Top', select='vertices in (y > 0.499)', kind='facet')
        # region_3 = domain.create_region(name='Bottom', select='vertices in (y < -0.499)', kind='facet')
        # region_4 = domain.create_region(name='Point', select='vertices in (x < -0.49) & (y < -0.49)', kind='vertex')
        
        # ebc_1 = EssentialBC(name='Right', region=region_0, dofs={'u.0' : 0.0,'u.1' : u_Right}) # 2.15
        # ebc_2 = EssentialBC(name='Left', region=region_1, dofs={'u.0' : 0.0,'u.1' : u_Left}) # -2.25
        # ebc_3 = EssentialBC(name='Bottom', region=region_3, dofs={'u.0' : u_Bottom,'u.1' : 0.0}) # -2.0
        # ebc_4 = EssentialBC(name='Top', region=region_2, dofs={'u.0' : u_Top,'u.1' : 0.0}) # 2.5
        # ebc_5 = EssentialBC(name='Point', region=region_4, dofs={'p.0' : 1e-9})
        
        # viscosity = Material(name='viscosity', value=viscosity) # 1e-3
        # density = Material(name='density', value=density) # 30.0
        
        # variable_1 = FieldVariable('u', 'unknown', field_1)
        # variable_2 = FieldVariable(name='v', kind='test', field=field_1, primary_var_name='u')
        # variable_3 = FieldVariable(name='p', kind='unknown', field=field_2)
        # variable_4 = FieldVariable(name='q', kind='test', field=field_2, primary_var_name='p')
        
        # integral_1 = Integral('i1', order=2) # 2 
        # integral_2 = Integral('i2', order=2) # 3
        
        # t1 = Term.new(name='dw_div_grad(viscosity.value, v, u)',
        #               integral=integral_2, region=omega, viscosity=viscosity, v=variable_2, u=variable_1)
        # # t2 = Term.new(name='dw_convect(v, u)',
        # #               integral=integral_2, region=omega, v=variable_2, u=variable_1)
        # t3 = Term.new(name='dw_stokes(density.value, v, p)',
        #               integral=integral_1, density=density, region=omega, v=variable_2, p=variable_3)
        # t4 = Term.new(name='dw_stokes(u, q)',
        #               integral=integral_1, region=omega, u=variable_1, q=variable_4)
        # eq1 = Equation('balance', t1-t3)
        # eq2 = Equation('incompressibility', t4)
        # eqs = Equations([eq1,eq2])
        
        # ls = ScipySuperLU({})
        # nls_status = IndexedStruct()
        # nls = Newton({'i_max' : 2000, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'or', 'macheps' : 1e-16, 'lin_red' : 1e-4, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-2}, lin_solver=ls, status=nls_status)
        # pb = Problem('Navier-Stokes', equations=eqs)
        # pb.set_bcs(ebcs=Conditions([ebc_1, ebc_2, ebc_3, ebc_4, ebc_5]))
        # pb.set_solver(nls)
        # status = IndexedStruct()
        # state = pb.solve(status=status, save_results=True)
        
        # out = state.create_output_dict()
        # pb.save_state('Navier_Stokes.vtk', out=out)
        
        # prb = Probe(out,pb.domain.mesh)
        
        # prb.add_points_probe('center', [[0.0,0.0,0.0]])
        # pars,vals = prb.__call__('center','u')
        
        # resultant = np.sqrt(vals[:,0]**2+vals[:,1]**2)
        
        # return resultant
        
        visc = viscosity
        dens = density
        utop = u_Top
        ubot = u_Bottom
        uright = u_Right
        uleft = u_Left
        
        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S.i', 'r') 
        Lines = file1.readlines()
        Lines[109] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
        Lines[76] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
        Lines[82] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
        Lines[88] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
        Lines[94] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"
        
        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S.i', 'w') 
        file1.writelines(Lines) 
        file1.close() 
        
        os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
        os.system('/Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i S.i')
        
        path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            Samp1 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                    Samp1.append(float(row[2]))
                count = count + 1
        
        resultant_S = np.sqrt(Samp0[1]**2+Samp1[1]**2)
        return resultant_S
    
    def Stokes1(self, viscosity, density, u_Bottom, u_Top, u_Left, u_Right):
        
        # mesh = Mesh.from_file(data_dir + '/meshes/3d/fluid_lid.inp')
        # domain = FEDomain('domain', mesh)
        
        # omega = domain.create_region('Omega', 'all')
        # field_1 = Field.from_args(name='3_velocity', dtype=nm.float64, shape=2, region=omega, approx_order=2)
        # field_2 = Field.from_args(name='pressure', dtype=nm.float64, shape=1, region=omega, approx_order=1)
        
        # region_0 = domain.create_region(name='Right', select='vertices in (x > 0.499)', kind='facet')
        # region_1 = domain.create_region(name='Left', select='vertices in (x < -0.499)', kind='facet')
        # region_2 = domain.create_region(name='Top', select='vertices in (y > 0.499)', kind='facet')
        # region_3 = domain.create_region(name='Bottom', select='vertices in (y < -0.499)', kind='facet')
        # region_4 = domain.create_region(name='Point', select='vertices in (x < -0.49) & (y < -0.49)', kind='vertex')
        
        # ebc_1 = EssentialBC(name='Right', region=region_0, dofs={'u.0' : 0.0,'u.1' : u_Right}) # 2.15
        # ebc_2 = EssentialBC(name='Left', region=region_1, dofs={'u.0' : 0.0,'u.1' : u_Left}) # -2.25
        # ebc_3 = EssentialBC(name='Bottom', region=region_3, dofs={'u.0' : u_Bottom,'u.1' : 0.0}) # -2.0
        # ebc_4 = EssentialBC(name='Top', region=region_2, dofs={'u.0' : u_Top,'u.1' : 0.0}) # 2.5
        # ebc_5 = EssentialBC(name='Point', region=region_4, dofs={'p.0' : 1e-9})
        
        # viscosity = Material(name='viscosity', value=viscosity) # 1e-3
        # density = Material(name='density', value=density) # 30.0
        
        # variable_1 = FieldVariable('u', 'unknown', field_1)
        # variable_2 = FieldVariable(name='v', kind='test', field=field_1, primary_var_name='u')
        # variable_3 = FieldVariable(name='p', kind='unknown', field=field_2)
        # variable_4 = FieldVariable(name='q', kind='test', field=field_2, primary_var_name='p')
        
        # integral_1 = Integral('i1', order=2) # 2 
        # integral_2 = Integral('i2', order=2) # 3
        
        # t1 = Term.new(name='dw_div_grad(viscosity.value, v, u)',
        #               integral=integral_2, region=omega, viscosity=viscosity, v=variable_2, u=variable_1)
        # # t2 = Term.new(name='dw_convect(v, u)',
        # #               integral=integral_2, region=omega, v=variable_2, u=variable_1)
        # t3 = Term.new(name='dw_stokes(density.value, v, p)',
        #               integral=integral_1, density=density, region=omega, v=variable_2, p=variable_3)
        # t4 = Term.new(name='dw_stokes(u, q)',
        #               integral=integral_1, region=omega, u=variable_1, q=variable_4)
        # eq1 = Equation('balance', t1-t3)
        # eq2 = Equation('incompressibility', t4)
        # eqs = Equations([eq1,eq2])
        
        # ls = ScipySuperLU({})
        # nls_status = IndexedStruct()
        # nls = Newton({'i_max' : 2000, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'or', 'macheps' : 1e-16, 'lin_red' : 1e-4, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-2}, lin_solver=ls, status=nls_status)
        # pb = Problem('Navier-Stokes', equations=eqs)
        # pb.set_bcs(ebcs=Conditions([ebc_1, ebc_2, ebc_3, ebc_4, ebc_5]))
        # pb.set_solver(nls)
        # status = IndexedStruct()
        # state = pb.solve(status=status, save_results=True)
        
        # out = state.create_output_dict()
        # pb.save_state('Navier_Stokes.vtk', out=out)
        
        # prb = Probe(out,pb.domain.mesh)
        
        # prb.add_points_probe('center', [[0.0,0.0,0.0]])
        # pars,vals = prb.__call__('center','u')
        
        # resultant = np.sqrt(vals[:,0]**2+vals[:,1]**2)
        
        # return resultant
        
        visc = viscosity
        dens = density
        utop = u_Top
        ubot = u_Bottom
        uright = u_Right
        uleft = u_Left
        
        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S1.i', 'r') 
        Lines = file1.readlines()
        Lines[109] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
        Lines[76] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
        Lines[82] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
        Lines[88] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
        Lines[94] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"
        
        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S1.i', 'w') 
        file1.writelines(Lines) 
        file1.close() 
        
        os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
        os.system('/Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i S1.i')
        
        path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S1.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            Samp1 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                    Samp1.append(float(row[2]))
                count = count + 1
        
        resultant_S = np.sqrt(Samp0[1]**2+Samp1[1]**2)
        return resultant_S