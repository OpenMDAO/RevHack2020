"""
OpenMDAO model file prototyping the eCRM analysis that can be used to calculate stability derivatives.
"""
import pickle

import numpy as np
import matplotlib.pylab as plt

import openmdao.api as om
from openaerostruct.aerodynamics.aero_groups import AeroPoint

from wing_geom_changes import VSPeCRM


#Read baseline mesh
with open('baseline_meshes.pkl', "rb") as f:
    meshes = pickle.load(f)

# Define data for surfaces
wing_surface = {
    # Wing definition
    'name' : 'wing',        # name of the surface
    'symmetry' : True,      # if true, model one half of wing
                            # reflected across the plane y = 0
    'S_ref_type' : 'wetted', # how we compute the wing area,
                             # can be 'wetted' or 'projected'
    'fem_model_type': 'tube',
    'twist_cp': np.array([0]),
    'mesh': meshes['wing_mesh'].reshape((45, 33, 3), order='F'),

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

surfaces = [wing_surface]

prob = om.Problem()
model = prob.model

#model.add_subsystem('vsp', VSPeCRM(horiz_tail_name="Tail",
                                   #vert_tail_name="VerticalTail",
                                   #wing_name="Wing"))

# Create the aero point group, which contains the actual aerodynamic
# analyses
aero_group = AeroPoint(surfaces=surfaces)
prob.model.add_subsystem('aero', aero_group,
                         promotes_inputs=['v', 'alpha', 'Mach_number', 're', 'rho', 'cg'])


prob.setup()

# Initial conditions
prob.set_val('v', val=248.136, units='m/s')
prob.set_val('alpha', val=0.0, units='deg')
prob.set_val('Mach_number', val=0.1)                   # 70 mph approx. TODO: make exact
prob.set_val('re', val=1.0e6, units='1/m')
prob.set_val('rho', val=0.38, units='kg/m**3')
prob.set_val('cg', val=np.zeros((3)), units='m')

prob.run_model()

print('done')