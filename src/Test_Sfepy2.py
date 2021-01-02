#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:03:14 2020

@author: dhulls
"""

from __future__ import print_function
from __future__ import absolute_import
from argparse import ArgumentParser
import numpy as nm

import sys
sys.path.append('.')

from sfepy.base.base import IndexedStruct
from sfepy.discrete import (FieldVariable, Material, Integral, Function,
                            Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.postprocess.viewer import Viewer
from sfepy.mechanics.matcoefs import stiffness_from_lame
import numpy as np


def shift_u_fun(ts, coors, bc=None, problem=None, shift=0.0):
    """
    Define a displacement depending on the y coordinate.
    """
    val = shift * coors[:,1]**2

    return val

helps = {
    'show' : 'show the results figure',
}

# def main():
from sfepy import data_dir

k = 1e5 # Elastic plane stiffness for positive penetration.
f0 = 1e2 # Force at zero penetration.
dn = 0.2 # x or y component magnitude of normals.
ds = 0.25 # Boundary polygon size in horizontal directions.
az = 0.4 # Anchor z coordinate.

parser = ArgumentParser()
parser.add_argument('--version', action='version', version='%(prog)s')
parser.add_argument('-s', '--show',
                    action="store_true", dest='show',
                    default=False, help=helps['show'])
options = parser.parse_args()

mesh = Mesh.from_file(data_dir + '/meshes/3d/cube_medium_hexa.mesh')
domain = FEDomain('domain', mesh)

omega = domain.create_region('Omega', 'all')
Bottom = domain.create_region('Bottom',
                              'vertices in z < %.10f' % -0.499,
                              'facet')
Top = domain.create_region('Top',
                              'vertices in z > %.10f' % 0.499,
                              'facet')

field = Field.from_args('fu', nm.float64, 'vector', omega,
                        approx_order=2)

u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name='u')

m = Material('m', D=stiffness_from_lame(dim=3, lam=5.769, mu=3.846))

cs0 = Material('cs',f=[k, f0],n=[dn, 0.0, 1.0],a=[0.0, 0.0, az],bs=[[0.0, 0.0, az],
                 [-ds, -ds, az],
                 [-ds, ds, az]])

cs1 = Material('cs',f=[k, f0],n=[-dn, 0.0, 1.0],a=[0.0, 0.0, az],bs=[[0.0, 0.0, az],
                 [ds, -ds, az],
                 [ds, ds, az]])

cs2 = Material('cs',f=[k, f0],n=[0.0, dn, 1.0],a=[0.0, 0.0, az],bs=[[0.0, 0.0, az],
                 [-ds, -ds, az],
                 [ds, -ds, az]])

cs3 = Material('cs',f=[k, f0],n=[0.0, -dn, 1.0],a=[0.0, 0.0, az],bs=[[0.0, 0.0, az],
                 [-ds, ds, az],
                 [ds, ds, az]])

integral = Integral('i', order=3)
integral1 = Integral('i', order=2)

t1 = Term.new('dw_lin_elastic(m.D, v, u)',
              integral, omega, m=m, v=v, u=u)
t2 = Term.new('dw_contact_plane(cs.f, cs.n, cs.a, cs.bs, v, u)', integral1, Top, cs=cs0, v=v, u=u)
t3 = Term.new('dw_contact_plane(cs.f, cs.n, cs.a, cs.bs, v, u)', integral1, Top, cs=cs1, v=v, u=u)
t4 = Term.new('dw_contact_plane(cs.f, cs.n, cs.a, cs.bs, v, u)', integral1, Top, cs=cs2, v=v, u=u)
t5 = Term.new('dw_contact_plane(cs.f, cs.n, cs.a, cs.bs, v, u)', integral1, Top, cs=cs3, v=v, u=u)
eq = Equation('balance', t1 + t2 + t3 + t4 + t5)
eqs = Equations([eq])

fix_u = EssentialBC('fix_u', Bottom, {'u.all' : 0.0})

ls = ScipyDirect({})

nls_status = IndexedStruct()
nls = Newton({}, lin_solver=ls, status=nls_status)

pb = Problem('elasticity', equations=eqs)
pb.save_regions_as_groups('regions')

pb.set_bcs(ebcs=Conditions([fix_u]))

pb.set_solver(nls)

status = IndexedStruct()
state = pb.solve(status=status)

print('Nonlinear solver status:\n', nls_status)
print('Stationary solver status:\n', status)

pb.save_state('linear_elasticity.vtk', state)

# if options.show:
view = Viewer('linear_elasticity.vtk')
view(vector_mode='warp_norm', rel_scaling=2,
      is_scalar_bar=True, is_wireframe=True)