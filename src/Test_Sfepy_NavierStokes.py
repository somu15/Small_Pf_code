#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 09:33:53 2020

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
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
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

mesh = Mesh.from_file(data_dir + '/meshes/3d/fluid_mesh.inp')
domain = FEDomain('domain', mesh)

omega = domain.create_region('Omega', 'all')
field_1 = Field.from_args(name='3_velocity', dtype=nm.float64, shape=3, region=omega, approx_order=1)
field_2 = Field.from_args(name='pressure', dtype=nm.float64, shape=1, region=omega, approx_order=1)

region_0 = domain.create_region(name='Walls1', select='vertices in (y < -0.049)', kind='facet')
region_1 = domain.create_region(name='Walls2', select='vertices in (y > 0.049)', kind='facet')
region_2 = domain.create_region(name='Inlet', select='vertices in (x < -0.499)', kind='facet')
region_3 = domain.create_region(name='Outlet', select='vertices in (x > -0.499)', kind='facet')

ebc_1 = EssentialBC(name='Walls1', region=region_0, dofs={'u.[0,1,2]' : 0.0})
ebc_2 = EssentialBC(name='Walls2', region=region_1, dofs={'u.[0,1,2]' : 0.0})
ebc_3 = EssentialBC(name='Inlet', region=region_2, dofs={'u.0' : 1.0, 'u.[1,2]' : 0.0})
ebc_4 = EssentialBC(name='Outlet', region=region_3, dofs={'p':0.0, 'u.[1,2]' : 0.0})

viscosity = Material(name='viscosity', value=1.25e-3)

variable_1 = FieldVariable('u', 'unknown', field_1)
variable_2 = FieldVariable(name='v', kind='test', field=field_1, primary_var_name='u')
variable_3 = FieldVariable(name='p', kind='unknown', field=field_2)
variable_4 = FieldVariable(name='q', kind='test', field=field_2, primary_var_name='p')

integral_1 = Integral('i1', order=2)
integral_2 = Integral('i2', order=3)

t1 = Term.new(name='dw_div_grad(viscosity.value, v, u)',
              integral=integral_2, region=omega, viscosity=viscosity, v=variable_2, u=variable_1)
t2 = Term.new(name='dw_convect(v, u)',
              integral=integral_2, region=omega, v=variable_2, u=variable_1)
t3 = Term.new(name='dw_stokes(v, p)',
              integral=integral_1, region=omega, v=variable_2, p=variable_3)
t4 = Term.new(name='dw_stokes(u, q)',
              integral=integral_1, region=omega, u=variable_1, q=variable_4)
eq1 = Equation('balance', t1+t2-t3)
eq2 = Equation('incompressibility', t4)
eqs = Equations([eq1,eq2])

ls = ScipyDirect({})
nls_status = IndexedStruct()
nls = Newton({'i_max' : 20, 'eps_a' : 1e-8, 'eps_r' : 1.0, 'macheps' : 1e-16, 'lin_red' : 1e-2, 'ls_red' : 0.1, 'ls_red_warp' : 0.001, 'ls_on' : 0.99999, 'ls_min' : 1e-5, 'check' : 0, 'delta' : 1e-6}, lin_solver=ls, status=nls_status)
pb = Problem('Navier-Stokes', equations=eqs)
pb.set_bcs(ebcs=Conditions([ebc_1, ebc_2, ebc_3]))
pb.set_solver(nls)
status = IndexedStruct()
state = pb.solve(status=status, save_results=True)

out = state.create_output_dict()
pb.save_state('Navier_Stokes.vtk', out=out)

view = Viewer('Navier_Stokes.vtk')
view(rel_scaling=2,
      is_scalar_bar=True, is_wireframe=True)