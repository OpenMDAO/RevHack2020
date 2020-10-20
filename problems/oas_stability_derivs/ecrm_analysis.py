"""
OpenMDAO model file prototyping the eCRM analysis that can be used to calculate stability derivatives.
"""
import pickle

import numpy as np
import matplotlib.pylab as plt

import openmdao.api as om
from openaerostruct.aerodynamics.aero_groups import AeroPoint

from vsp_eCRM import VSPeCRM


#Read baseline mesh
with open('baseline_meshes_reduced.pkl', "rb") as f:
    meshes = pickle.load(f)

wing_mesh = meshes['wing_mesh']
horiz_tail_mesh = meshes['horiz_tail_mesh']
vert_tail_mesh = meshes['vert_tail_mesh']

# Define data for surfaces
wing_surface = {
    # Wing definition
    'name' : 'wing',        # name of the surface
    'symmetry' : True,      # if true, model one half of wing
                            # reflected across the plane y = 0
    'S_ref_type' : 'wetted', # how we compute the wing area,
                             # can be 'wetted' or 'projected'
    'fem_model_type': 'tube',
    'twist_cp': np.zeros((1)),
    'mesh': wing_mesh,

    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    'CL0' : 0.0,            # CL of the surface at alpha=0
    'CD0' : 0.015,          # CD of the surface at alpha=0

    # Airfoil properties for viscous drag calculation
    'k_lam' : 0.05,         # percentage of chord with laminar
                            # flow, used for viscous drag
    't_over_c': 0.15,  # thickness over chord ratio (NACA0015)
    'c_max_t': 0.303,  # chordwise location of maximum (NACA0015)
                            # thickness
    'with_viscous' : True,
    'with_wave' : False,     # if true, compute wave drag

}

# TODO - need real data for horiz tail.
horiz_tail_surface = {
    # Wing definition
    'name': 'horiz_tail',  # name of the surface
    'symmetry': True,  # if true, model one half of wing
    # reflected across the plane y = 0
    'S_ref_type': 'wetted',  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    'twist_cp': np.zeros((1)),
    'twist_cp_dv': False,
    'mesh': horiz_tail_mesh,
    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    'CL0': 0.0,  # CL of the surface at alpha=0
    'CD0': 0.01,  # CD of the surface at alpha=0
    'fem_origin': 0.35,
    # Airfoil properties for viscous drag calculation
    'k_lam': 0.05,  # percentage of chord with laminar
    # flow, used for viscous drag
    't_over_c': 0.15,  # thickness over chord ratio (NACA0015)
    'c_max_t': 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    'with_viscous': True,  # if true, compute viscous drag
    'with_wave': False,
}

vert_tail_surface = {
    # Wing definition
    'name': 'vert_tail',  # name of the surface
    'symmetry': True,  # if true, model one half of wing
    # reflected across the plane y = 0
    'S_ref_type': 'wetted',  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    'twist_cp': np.zeros((1)),
    'twist_cp_dv': False,
    'mesh': vert_tail_mesh,
    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    'CL0': 0.0,  # CL of the surface at alpha=0
    'CD0': 0.01,  # CD of the surface at alpha=0
    'fem_origin': 0.35,
    # Airfoil properties for viscous drag calculation
    'k_lam': 0.05,  # percentage of chord with laminar
    # flow, used for viscous drag
    't_over_c': 0.15,  # thickness over chord ratio (NACA0015)
    'c_max_t': 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    'with_viscous': True,  # if true, compute viscous drag
    'with_wave': False,
}

surfaces = [wing_surface, horiz_tail_surface, vert_tail_surface]

prob = om.Problem()
model = prob.model

# Using manual indepvarcomp because of some promote errors in OAS if I omit it.
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output('v', val=248.136, units='m/s')
indep_var_comp.add_output('alpha', val=0.0, units='deg')
indep_var_comp.add_output('Mach_number', val=0.1)                   # 70 mph approx. TODO: make exact
indep_var_comp.add_output('re', val=1.0e6, units='1/m')
indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
indep_var_comp.add_output('cg', val=np.zeros((3)), units='m')
prob.model.add_subsystem('prob_vars', indep_var_comp, promotes=['*'])

model.add_subsystem('vsp', VSPeCRM(horiz_tail_name="Tail",
                                   vert_tail_name="VerticalTail",
                                   wing_name="Wing"))

prob.model.connect('vsp.wing_mesh', 'aero.wing.def_mesh')
prob.model.connect('vsp.wing_mesh', 'aero.aero_states.wing_def_mesh')
prob.model.connect('vsp.horiz_tail_mesh', 'aero.horiz_tail.def_mesh')
prob.model.connect('vsp.horiz_tail_mesh', 'aero.aero_states.horiz_tail_def_mesh')
prob.model.connect('vsp.vert_tail_mesh', 'aero.vert_tail.def_mesh')
prob.model.connect('vsp.vert_tail_mesh', 'aero.aero_states.vert_tail_def_mesh')

# Create the aero point group, which contains the actual aerodynamic
# analyses
aero_group = AeroPoint(surfaces=surfaces)
prob.model.add_subsystem('aero', aero_group,
                         promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'cg'])


prob.setup()

# Initial conditions
#prob.set_val('v', val=248.136, units='m/s')
#prob.set_val('alpha', val=0.0, units='deg')
#prob.set_val('Mach_number', val=0.1)                   # 70 mph approx. TODO: make exact
#prob.set_val('re', val=1.0e6, units='1/m')
#prob.set_val('rho', val=0.38, units='kg/m**3')
#prob.set_val('cg', val=np.zeros((3)), units='m')

# This stuff only when we don't have the geo turned on.
#prob.set_val('aero.wing.def_mesh', wing_mesh)
#prob.set_val('aero.aero_states.wing_def_mesh', wing_mesh)

prob.run_model()

wrt = ['alpha', 'vsp.wing_cord', 'vsp.vert_tail_area', 'vsp.horiz_tail_area']
of = ['aero.CL', 'aero.CD', 'aero.CM']

totals = prob.compute_totals(of=of, wrt=wrt)
print(totals)
print('done')