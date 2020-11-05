"""
OpenMDAO optimization of eCRM design subject to stability derivative constraints.

This model is best run in MPI with 3 processors.
"""
import pickle

import numpy as np

import openmdao.api as om
from openaerostruct.utils.constants import grav_constant

from ecrm_comp_with_stability_derivs import ECRM


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

    'thickness_cp' : np.array([.1, .25]),

    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    'CL0' : 0.7,            # CL of the surface at alpha=0
    'CD0' : 0.015,          # CD of the surface at alpha=0

    # Airfoil properties for viscous drag calculation
    'k_lam' : 0.05,         # percentage of chord with laminar
                            # flow, used for viscous drag
    't_over_c_cp': np.array([0.15]),  # thickness over chord ratio (NACA0015)
    'c_max_t': 0.303,  # chordwise location of maximum (NACA0015)
                            # thickness
    'with_viscous' : True,
    'with_wave' : False,     # if true, compute wave drag

    # Structural values are based on aluminum 7075
    'E' : 70.e9,            # [Pa] Young's modulus of the spar
    'G' : 30.e9,            # [Pa] shear modulus of the spar
    'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
    'mrho' : 2.78e3,          # [kg/m^3] material density
    'fem_origin' : 0.35,    # normalized chordwise location of the spar
    'wing_weight_ratio' : 1.25,
    'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
    'distributed_fuel_weight' : False,
    # Constraints
    'exact_failure_constraint' : False, # if false, use KS function
}

# TODO - need real data for horiz tail.
horiz_tail_surface = {
    # Wing definition
    'name': 'horiz_tail',  # name of the surface
    'symmetry': True,  # if true, model one half of wing
    # reflected across the plane y = 0
    'S_ref_type': 'wetted',  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    'fem_model_type': 'tube',
    'twist_cp': np.zeros((1)),
    'twist_cp_dv': False,

    'mesh': horiz_tail_mesh,

    'thickness_cp' : np.array([.01, .02]),

    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    'CL0': 0.0,  # CL of the surface at alpha=0
    'CD0': 0.0,  # CD of the surface at alpha=0
    # Airfoil properties for viscous drag calculation
    'k_lam': 0.05,  # percentage of chord with laminar
    # flow, used for viscous drag
    't_over_c_cp': np.array([0.15]),  # thickness over chord ratio (NACA0015)
    'c_max_t': 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    'with_viscous': True,  # if true, compute viscous drag
    'with_wave': False,

    # Structural values are based on aluminum 7075
    'E' : 70.e9,            # [Pa] Young's modulus of the spar
    'G' : 30.e9,            # [Pa] shear modulus of the spar
    'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
    'mrho' : 2.78e3,          # [kg/m^3] material density
    'fem_origin' : 0.35,    # normalized chordwise location of the spar
    'wing_weight_ratio' : 1.25,
    'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
    'distributed_fuel_weight' : False,
    # Constraints
    'exact_failure_constraint' : False, # if false, use KS function
}

vert_tail_surface = {
    # Wing definition
    'name': 'vert_tail',  # name of the surface
    'symmetry': False,  # if true, model one half of wing
    # reflected across the plane y = 0
    'S_ref_type': 'wetted',  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    'fem_model_type': 'tube',
    'twist_cp': np.zeros((1)),
    'twist_cp_dv': False,

    'mesh': vert_tail_mesh,

    'thickness_cp' : np.array([.01, .02]),

    # Aerodynamic performance of the lifting surface at
    # an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    'CL0': 0.0,  # CL of the surface at alpha=0
    'CD0': 0.0,  # CD of the surface at alpha=0
    # Airfoil properties for viscous drag calculation
    'k_lam': 0.05,  # percentage of chord with laminar
    # flow, used for viscous drag
    't_over_c_cp': np.array([0.15]),  # thickness over chord ratio (NACA0015)
    'c_max_t': 0.303,  # chordwise location of maximum (NACA0015)
    # thickness
    'with_viscous': True,  # if true, compute viscous drag
    'with_wave': False,

    # Structural values are based on aluminum 7075
    'E' : 70.e9,            # [Pa] Young's modulus of the spar
    'G' : 30.e9,            # [Pa] shear modulus of the spar
    'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
    'mrho' : 2.78e3,          # [kg/m^3] material density
    'fem_origin' : 0.35,    # normalized chordwise location of the spar
    'wing_weight_ratio' : 1.25,
    'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
    'distributed_fuel_weight' : False,
    # Constraints
    'exact_failure_constraint' : False, # if false, use KS function
}

#vels = np.array([70.0])                    # Test run
#vels = np.array([150.0, 70.0, 200.0])      # Actual full run
vels = np.array([150.0, 155.0, 160.0])     # Demonstration run
num_nodes = len(vels)

prob = om.Problem()
model = prob.model

design_inputs = ['wing_cord', 'vert_tail_area', 'horiz_tail_area']
common_settings = ['beta', 're', 'rho', 'CT', 'R', 'W0', 'load_factor', 'speed_of_sound', 'empty_cg']

model.add_subsystem('ecrm', ECRM(wing_surface=wing_surface,
                                 horiz_tail_surface=horiz_tail_surface,
                                 vert_tail_surface=vert_tail_surface,
                                 num_nodes=num_nodes),
                    promotes_inputs=design_inputs + common_settings)

# Objective: Maximize L/D @ 150 mph
model.add_subsystem('l_over_d', om.ExecComp('val = -CL / CD'))
model.connect('ecrm.CL', 'l_over_d.CL', src_indices=0)
model.connect('ecrm.CD', 'l_over_d.CD', src_indices=0)
model.add_objective('l_over_d.val')

# Constraint: -CM_α/CL_α > 0.0
con_alpha = om.ExecComp('val = -CMa / CLa',
                        CMa=np.ones(num_nodes), CLa=np.ones(num_nodes), val=np.ones(num_nodes))
model.add_subsystem('con_alpha', con_alpha)
model.connect('ecrm.CM_alpha', 'con_alpha.CMa')
model.connect('ecrm.CL_alpha', 'con_alpha.CLa')
model.add_constraint('con_alpha.val', lower=0.0)

# Constraint: CN_β > 0.0
model.add_constraint('ecrm.CN_beta', lower=0.0)

# Constraint: CL < 1.3
model.add_constraint('ecrm.CL', upper=1.3)

# Constraint: CL = W/qS
model.add_constraint('ecrm.L_equals_W', equals=0.0)

# Design Variables
model.add_design_var('wing_cord', lower=20.0, upper=100, ref=59.0)
model.add_design_var('vert_tail_area', lower=1000.0, ref=2295.0)
model.add_design_var('horiz_tail_area', lower=1000.0, ref=6336.0)
model.add_design_var('ecrm.alpha', lower=-2.0, upper=22.0, ref=5.0)

prob.driver = om.pyOptSparseDriver()
prob.driver.options['optimizer'] = "SNOPT"
prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs']
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-3
prob.driver.opt_settings['Major optimality tolerance'] = 1e-5

prob.setup()

# Set Constant Initial Conditions
prob.set_val('beta', 0.0, units='deg')
prob.set_val('re', 1.0e6, units='1/m')
prob.set_val('rho', 1.225, units='kg/m**3')
#prob.set_val('rho', 1.0, units='kg/m**3')
#prob.set_val('rho', 0.38, units='kg/m**3')
prob.set_val('CT', grav_constant * 17.e-6, units='1/s')
prob.set_val('R', 50.0, units='km')
prob.set_val('W0', 1000.0,  units='kg')
prob.set_val('load_factor', 1.)
prob.set_val('speed_of_sound', 767.0, units='mi/h')
prob.set_val('empty_cg', np.array([262.614, 0.0, 115.861]), units='cm')

# Set Airspeeds for all models
prob.set_val('ecrm.v', vels, units='mi/h')
prob.set_val('ecrm.Mach_number', vels/767.0)
prob.set_val('ecrm.alpha', np.array([13.0]))#, 7.0, 2.0]))

# Initial VSP Design
prob.set_val('wing_cord', 59.05128, units='cm')
prob.set_val('vert_tail_area', 2295.0, units='cm**2')
prob.set_val('horiz_tail_area', 6336.0, units='cm**2')

#prob.run_model()
#z=prob.check_totals()

prob.run_driver()

prob.list_problem_vars()

for var in ['ecrm.alpha', 'wing_cord', 'vert_tail_area', 'horiz_tail_area']:
    print(var, prob.get_val(var))

print('done')