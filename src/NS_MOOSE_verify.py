#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:02:53 2021

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
from sfepy.discrete.conditions import Conditions, EssentialBC, InitialCondition
from sfepy.solvers.ls import ScipyDirect, ScipySuperLU, ScipyIterative
from sfepy.solvers.nls import Newton, ScipyBroyden, PETScNonlinearSolver
from sfepy.solvers.oseen import Oseen
from sfepy.solvers.oseen import StabilizationFunction
from sfepy.postprocess.viewer import Viewer
from sfepy.postprocess.probes_vtk import ProbeFromFile, Probe
import numpy as np

helps = {
    'show' : 'show the results figure',
}

from sfepy import data_dir

parser = ArgumentParser()
parser.add_argument('--version', action='version', version='%(prog)s')
parser.add_argument('-s', '--show',
                    action="store_true", dest='show',
                    default=False, help=helps['show'])
options = parser.parse_args()

# ############## Navier-Stokes 2D NEWTON solver #################

mesh = Mesh.from_file(data_dir + '/meshes/3d/fluid_lid.inp')
domain = FEDomain('domain', mesh)

omega = domain.create_region('Omega', 'all')
field_1 = Field.from_args(name='3_velocity', dtype=nm.float64, shape=2, region=omega, approx_order=2)
field_2 = Field.from_args(name='pressure', dtype=nm.float64, shape=1, region=omega, approx_order=1)

region_0 = domain.create_region(name='Right', select='vertices in (x > 0.499)', kind='facet')
region_1 = domain.create_region(name='Left', select='vertices in (x < -0.499)', kind='facet')
region_2 = domain.create_region(name='Top', select='vertices in (y > 0.499)', kind='facet')
region_3 = domain.create_region(name='Bottom', select='vertices in (y < -0.499)', kind='facet')
region_4 = domain.create_region(name='Point', select='vertices in (x < -0.49) & (y < -0.49)', kind='vertex')

ebc_1 = EssentialBC(name='Right', region=region_0, dofs={'u.0' : 0.0,'u.1' : 2.15})
ebc_2 = EssentialBC(name='Left', region=region_1, dofs={'u.0' : 0.0,'u.1' : -2.25})
ebc_3 = EssentialBC(name='Bottom', region=region_3, dofs={'u.0' : -2.0,'u.1' : 0.0})
ebc_4 = EssentialBC(name='Top', region=region_2, dofs={'u.0' : 2.5,'u.1' : 0.0})
ebc_5 = EssentialBC(name='Point', region=region_4, dofs={'p.0' : 1e-7})
# ebc_1 = EssentialBC(name='Right', region=region_0, dofs={'p.0' : 6500.0}) # , 'p' : 1.0
# ebc_2 = EssentialBC(name='Left', region=region_1, dofs={'p.0' : 5500.0})
# ebc_3 = EssentialBC(name='Top', region=region_2, dofs={'p.0' : 4000.0})
# ebc_4 = EssentialBC(name='Bottom', region=region_3, dofs={'p.0' : 4500.0})
# ic_1 = InitialCondition(name='IC',region=omega,dofs={'u.[0,1]' : 1e-5})

viscosity = Material(name='viscosity', value=1e-3)
density = Material(name='density', value=30.0)

variable_1 = FieldVariable('u', 'unknown', field_1)
variable_2 = FieldVariable(name='v', kind='test', field=field_1, primary_var_name='u')
variable_3 = FieldVariable(name='p', kind='unknown', field=field_2)
variable_4 = FieldVariable(name='q', kind='test', field=field_2, primary_var_name='p')

integral_1 = Integral('i1', order=2) # 2 
integral_2 = Integral('i2', order=2) # 3

t1 = Term.new(name='dw_div_grad(viscosity.value, v, u)',
              integral=integral_2, region=omega, viscosity=viscosity, v=variable_2, u=variable_1)
t2 = Term.new(name='dw_convect(v, u)',
              integral=integral_2, region=omega, v=variable_2, u=variable_1)
t3 = Term.new(name='dw_stokes(density.value, v, p)',
              integral=integral_1, density=density, region=omega, v=variable_2, p=variable_3)
t4 = Term.new(name='dw_stokes(u, q)',
              integral=integral_1, region=omega, u=variable_1, q=variable_4)
eq1 = Equation('balance', t1+t2-t3)
eq2 = Equation('incompressibility', t4)
eqs = Equations([eq1,eq2])

ls = ScipySuperLU({}) #   ScipyDirect({}) #  ScipyIterative({}) # 
nls_status = IndexedStruct()
nls = Newton({'i_max' : 2000, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'or', 'macheps' : 1e-16, 'lin_red' : 1e-4, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-2}, lin_solver=ls, status=nls_status)
# nls = ScipyBroyden({'i_max' : 500, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'and', 'macheps' : 1e-16, 'lin_red' : 0.9, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-6}, lin_solver=ls, status=nls_status)
# nls = Oseen({'i_max' : 2000, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'or', 'macheps' : 1e-16, 'lin_red' : 1e-4, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-2}, lin_solver=ls, status=nls_status)
pb = Problem('Navier-Stokes', equations=eqs)
pb.set_bcs(ebcs=Conditions([ebc_1, ebc_2, ebc_3, ebc_4, ebc_5]))
# pb.set_ics(ics=Conditions([ic_1]))
pb.set_solver(nls)
status = IndexedStruct()
state = pb.solve(status=status, save_results=True)

out = state.create_output_dict()
pb.save_state('Navier_Stokes.vtk', out=out)

prb = Probe(out,pb.domain.mesh)

prb.add_points_probe('center', [[0.0,0.0,0.0]])
pars,vals = prb.__call__('center','u')

view = Viewer('Navier_Stokes.vtk')
view(rel_scaling=0.2,
      is_scalar_bar=True, is_wireframe=False)

resultant = np.sqrt(vals[:,0]**2+vals[:,1]**2)

############## Navier-Stokes 2D Oseen solver #################

# mesh = Mesh.from_file(data_dir + '/meshes/3d/fluid_lid.inp')
# domain = FEDomain('domain', mesh)

# omega = domain.create_region('Omega', 'all')
# field_1 = Field.from_args(name='3_velocity', dtype=nm.float64, shape=2, region=omega, approx_order=2)
# field_2 = Field.from_args(name='pressure', dtype=nm.float64, shape=1, region=omega, approx_order=1)

# region_0 = domain.create_region(name='Right', select='vertices in (x > 0.499)', kind='facet')
# region_1 = domain.create_region(name='Left', select='vertices in (x < -0.499)', kind='facet')
# region_2 = domain.create_region(name='Top', select='vertices in (y > 0.499)', kind='facet')
# region_3 = domain.create_region(name='Bottom', select='vertices in (y < -0.499)', kind='facet')
# region_4 = domain.create_region(name='Point', select='vertices in (x < -0.499) & (y < -0.499)', kind='vertex')

# ebc_1 = EssentialBC(name='Right', region=region_0, dofs={'u.0' : 0.0,'u.1' : 1.0})
# ebc_2 = EssentialBC(name='Left', region=region_1, dofs={'u.0' : 0.0,'u.1' : -0.5})
# ebc_3 = EssentialBC(name='Bottom', region=region_3, dofs={'u.0' : -0.5,'u.1' : 0.0})
# ebc_4 = EssentialBC(name='Top', region=region_2, dofs={'u.0' : 1.0,'u.1' : 0.0})
# ebc_5 = EssentialBC(name='Point', region=region_4, dofs={'p.0' : 1e-7})
# # ebc_1 = EssentialBC(name='Right', region=region_0, dofs={'p.0' : 6500.0}) # , 'p' : 1.0
# # ebc_2 = EssentialBC(name='Left', region=region_1, dofs={'p.0' : 5500.0})
# # ebc_3 = EssentialBC(name='Top', region=region_2, dofs={'p.0' : 4000.0})
# # ebc_4 = EssentialBC(name='Bottom', region=region_3, dofs={'p.0' : 4500.0})
# # ic_1 = InitialCondition(name='IC',region=omega,dofs={'u.[0,1]' : 1e-5})

# viscosity = Material(name='viscosity', value=2.5e-3)
# density = Material(name='density', value=1.0)
# stabil = Material(name='stabil',value='stabil')

# variable_1 = FieldVariable('u', 'unknown', field_1)
# variable_2 = FieldVariable(name='v', kind='test', field=field_1, primary_var_name='u')
# variable_3 = FieldVariable(name='p', kind='unknown', field=field_2)
# variable_4 = FieldVariable(name='q', kind='test', field=field_2, primary_var_name='p')
# variable_5 = FieldVariable(name='b', kind='parameter', field=field_1, primary_var_name='u')

# integral_1 = Integral('i1', order=2) # 2 
# integral_2 = Integral('i2', order=2) # 3

# t1 = Term.new(name='dw_div_grad(viscosity.value, v, u)',
#               integral=integral_2, region=omega, viscosity=viscosity, v=variable_2, u=variable_1)
# t2 = Term.new(name='dw_convect(v, u)',
#               integral=integral_2, region=omega, v=variable_2, u=variable_1)
# t3 = Term.new(name='dw_stokes(density.value, v, p)',
#               integral=integral_1, density=density, region=omega, v=variable_2, p=variable_3)
# t3_0 = Term.new(name='dw_st_grad_div(stabil.gamma, v, u)',
#               integral=integral_1, stabil=stabil, region=omega, v=variable_2, u=variable_1)
# t3_1 = Term.new(name='dw_st_supg_c(stabil.delta, v, b, u)',
#               integral=integral_1, stabil=stabil, region=omega, v=variable_2, b=variable_5, u=variable_1)
# t3_2 = Term.new(name='dw_st_supg_p(stabil.delta, v, b, p)',
#               integral=integral_1, stabil=stabil, region=omega, v=variable_2, b=variable_5, p=variable_3)
# t4 = Term.new(name='dw_stokes(u, q)',
#               integral=integral_1, region=omega, u=variable_1, q=variable_4)
# eq1 = Equation('balance', t1+t2-t3+t3_1+t3_2)
# eq2 = Equation('incompressibility', t4)
# eqs = Equations([eq1,eq2])

# problem = Problem('Navier-Stokes', equations=eqs)
# ls = ScipyDirect({}) # ScipySuperLU({}) #     ScipyIterative({}) # 
# nls_status = IndexedStruct()
# # nls = Newton({'i_max' : 2000, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'or', 'macheps' : 1e-16, 'lin_red' : 1e-4, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-2}, lin_solver=ls, status=nls_status)
# # nls = ScipyBroyden({'i_max' : 500, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'and', 'macheps' : 1e-16, 'lin_red' : 0.9, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-6}, lin_solver=ls, status=nls_status)
# nls = Oseen({'stabil_mat' : 'stabil', 'adimensionalize' : False, 'check_navier_stokes_residual' : False, 'i_max' : 2000, 'eps_a' : 1e-8, 'eps_r' : 1e-4, 'eps_mode' : 'or', 'macheps' : 1e-16, 'lin_red' : 1e-4, 'ls_red' : 1e-3, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-16, 'check' : 0, 'delta' : 1e-2}, lin_solver=ls, status=nls_status)
# problem.set_bcs(ebcs=Conditions([ebc_1, ebc_2, ebc_3, ebc_4, ebc_5]))
# # pb.set_ics(ics=Conditions([ic_1]))
# problem.set_solver(nls)
# status = IndexedStruct()
# state = problem.solve(status=status, save_results=True)

# out = state.create_output_dict()
# problem.save_state('Navier_Stokes.vtk', out=out)

# prb = Probe(out,problem.domain.mesh)

# # p0 = [-0.5,  -0.5, 0]
# # p1 = [-0.5, 0.5, 0]
# # prb.add_line_probe('Test', p0, p1, 100)
# # pars,vals = prb.__call__('Test','p')

# # def Compute_Max_Pressure(probe,num_points,interval):
# #     lens = np.arange(-0.5,0.5,interval)
# #     max_press = 0
# #     for ii in np.arange(0,len(lens),1):
# #         p0 = [lens[ii],  -0.5, 0]
# #         p1 = [lens[ii], 0.5, 0]
# #         prb.add_line_probe('Test', p0, p1, num_points)
# #         pars,vals = prb.__call__('Test','p')
# #         req = np.max(np.abs(vals))
# #         if max_press < req:
# #             max_press = req
# #     return max_press

# # Max_pressure = Compute_Max_Pressure(prb,500,0.005);

# view = Viewer('Navier_Stokes.vtk')
# view(rel_scaling=0.2,
#       is_scalar_bar=True, is_wireframe=False)

