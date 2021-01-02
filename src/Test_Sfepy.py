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

parser = ArgumentParser()
parser.add_argument('--version', action='version', version='%(prog)s')
parser.add_argument('-s', '--show',
                    action="store_true", dest='show',
                    default=False, help=helps['show'])
options = parser.parse_args()

# mesh = Mesh.from_file(data_dir + '/meshes/2d/rectangle_tri.mesh')
mesh = Mesh.from_file(data_dir + '/meshes/3d/cube_medium_hexa.mesh')
domain = FEDomain('domain', mesh)

# min_x, max_x = domain.get_mesh_bounding_box()[:,0]
# eps = 1e-8 * (max_x - min_x)
omega = domain.create_region('Omega', 'all')
# gamma1 = domain.create_region('Gamma1',
#                               'vertices in x < %.10f' % (min_x + eps),
#                               'facet')
Bottom = domain.create_region('Bottom',
                              'vertices in z < %.10f' % -0.499,
                              'facet')
# gamma2 = domain.create_region('Gamma2',
#                               'vertices in x > %.10f' % (max_x - eps),
#                               'facet')

Top = domain.create_region('Top',
                              'vertices in z > %.10f' % 0.499,
                              'facet')

field = Field.from_args('fu', nm.float64, 'vector', omega,
                        approx_order=3)

u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name='u')

# materials = {
#     'solid' : ({
#         'D': stiffness_from_lame(dim=3, lam=5.769, mu=3.846),
#     },),
#     'cs' : ({
#         'f' : [1e5, 1e-2],
#         '.c' : [0.0, 0.0, 1.2],
#         '.r' : 0.8,
#     },),
# }

# defK = materials['cs'][0]
# cs = ContactSphere(csc['.c'], csc['.r'])

m = Material('m', D=stiffness_from_lame(dim=3, lam=5.769, mu=3.846))
# f = Material('f', val=[[0.02], [0.01]])
# csf = Material('csf', val=[1e5, 1e-2])
# csc = Material('csc', val=[0.0, 0.0, 1.2])
# csr = Material('csr', val=0.8)

cs = Material('cs',f=[1e5, 1e-2],c=[0.0, 0.0, 1.2],r=0.8)

integral = Integral('i', order=3)
integral1 = Integral('i', order=2)

t1 = Term.new('dw_lin_elastic(m.D, v, u)',
              integral, omega, m=m, v=v, u=u)
t2 = Term.new('dw_contact_sphere(cs.f, cs.c, cs.r, v, u)', integral1, Top, cs=cs, v=v, u=u)
eq = Equation('balance', t1 + t2)
eqs = Equations([eq])

fix_u = EssentialBC('fix_u', Bottom, {'u.all' : 0.0})

# bc_fun = Function('shift_u_fun', shift_u_fun,
#                   extra_args={'shift' : 0.01})
# shift_u = EssentialBC('shift_u', gamma2, {'u.0' : bc_fun})

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