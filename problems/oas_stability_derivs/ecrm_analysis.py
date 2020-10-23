"""
OpenMDAO model file prototyping the eCRM analysis that can be used to calculate stability derivatives.
"""
import pickle

import numpy as np
import matplotlib.pylab as plt

import openmdao.api as om

from openaerostruct.integration.aerostruct_groups import AerostructPoint
from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.tube_group import TubeGroup
from openaerostruct.structures.wingbox_group import WingboxGroup
from openaerostruct.utils.constants import grav_constant

from vsp_eCRM import VSPeCRM


class AerostructGeometries(om.Group):
    """
    Modification of AerostructGeometry to use VSP.

    Structural analysis only happens on the wing.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        # Splinecomp for the thickness control points.
        for surface in surfaces:
            if 't_over_c_cp' in surface.keys():
                name = surface['name']
                n_cp = len(surface['t_over_c_cp'])
                ny = surface['mesh'].shape[1]
                x_interp = np.linspace(0., 1., int(ny-1))

                spline_comp = om.SplineComp(method='bsplines', x_interp_val=x_interp, num_cp=n_cp,
                                            interp_options={'order' : min(n_cp, 4)})

                self.add_subsystem(f'{name}_t_over_c_bsp', spline_comp,
                                   promotes_inputs=[('t_over_c_cp', f'{name}:t_over_c_cp')],
                                   promotes_outputs=[('t_over_c', f'{name}:t_over_c')])
                spline_comp.add_spline(y_cp_name='t_over_c_cp', y_interp_name='t_over_c',
                                       y_cp_val=surface['t_over_c_cp'])

        # VSP Geometry.
        self.add_subsystem('vsp', VSPeCRM(horiz_tail_name="Tail",
                                          vert_tail_name="VerticalTail",
                                          wing_name="Wing",
                                          reduced=True),
                           promotes_inputs=['wing_cord', 'vert_tail_area', 'horiz_tail_area'],
                           promotes_outputs=['wing_mesh', 'vert_tail_mesh', 'horiz_tail_mesh'])

        # Setting up the structural solve.
        for surface in surfaces:
            name = surface['name']
            sub = self.add_subsystem(name, om.Group())

            if surface['fem_model_type'] == 'tube':
                tube_promotes = []
                tube_inputs = []
                if 'thickness_cp' in surface.keys():
                    tube_promotes.append('thickness_cp')
                if 'radius_cp' not in surface.keys():
                    tube_inputs = ['mesh', 't_over_c']
                sub.add_subsystem('tube_group',
                                  TubeGroup(surface=surface, connect_geom_DVs=True),
                                  promotes_inputs=tube_inputs,
                                  promotes_outputs=['A', 'Iy', 'Iz', 'J', 'radius', 'thickness'] + tube_promotes)

            elif surface['fem_model_type'] == 'wingbox':
                wingbox_promotes = []
                if 'skin_thickness_cp' in surface.keys() and 'spar_thickness_cp' in surface.keys():
                    wingbox_promotes.append('skin_thickness_cp')
                    wingbox_promotes.append('spar_thickness_cp')
                    wingbox_promotes.append('skin_thickness')
                    wingbox_promotes.append('spar_thickness')
                elif 'skin_thickness_cp' in surface.keys() or 'spar_thickness_cp' in surface.keys():
                    raise NameError('Please have both skin and spar thickness as design variables, not one or the other.')

                sub.add_subsystem('wingbox_group',
                                  WingboxGroup(surface=surface),
                                  promotes_inputs=['mesh', 't_over_c'],
                                  promotes_outputs=['A', 'Iy', 'Iz', 'J', 'Qz', 'A_enc', 'A_int', 'htop', 'hbottom', 'hfront', 'hrear'] + wingbox_promotes)
            else:
                raise NameError('Please select a valid `fem_model_type` from either `tube` or `wingbox`.')

            if surface['fem_model_type'] == 'wingbox':
                promotes = ['A_int']
            else:
                promotes = []

            sub.add_subsystem('struct_setup',
                              SpatialBeamSetup(surface=surface),
                              promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J'] + promotes,
                              promotes_outputs=['nodes', 'local_stiff_transformed', 'structural_mass', 'cg_location', 'element_mass'])

            self.connect(f'{name}_mesh', [f'{name}.mesh'])
            self.connect(f'{name}:t_over_c', [f'{name}.t_over_c'])


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

    'thickness_cp' : np.array([.1, .2, .3]),

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
    't_over_c_cp': np.array([0.15]),  # thickness over chord ratio (NACA0015)
    'c_max_t': 0.303,  # chordwise location of maximum (NACA0015)
                            # thickness
    'with_viscous' : True,
    'with_wave' : False,     # if true, compute wave drag

    # Structural values are based on aluminum 7075
    'E' : 70.e9,            # [Pa] Young's modulus of the spar
    'G' : 30.e9,            # [Pa] shear modulus of the spar
    'yield' : 500.e6 / 2.5, # [Pa] yield stress divided by 2.5 for limiting case
    'mrho' : 3.e3,          # [kg/m^3] material density
    'fem_origin' : 0.35,    # normalized chordwise location of the spar
    'wing_weight_ratio' : 2.,
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

    'thickness_cp' : np.array([.1, .2]),

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
    'mrho' : 3.e3,          # [kg/m^3] material density
    'fem_origin' : 0.35,    # normalized chordwise location of the spar
    'wing_weight_ratio' : 2.,
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

    'thickness_cp' : np.array([.1, .2]),

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
    'mrho' : 3.e3,          # [kg/m^3] material density
    'fem_origin' : 0.35,    # normalized chordwise location of the spar
    'wing_weight_ratio' : 2.,
    'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
    'distributed_fuel_weight' : False,
    # Constraints
    'exact_failure_constraint' : False, # if false, use KS function
}

surfaces = [wing_surface, horiz_tail_surface, vert_tail_surface]
#surfaces = [wing_surface]

prob = om.Problem()
model = prob.model

# Using manual indepvarcomp because of some promote errors in OAS if I omit it.
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output('v', val=248.136, units='m/s')
indep_var_comp.add_output('alpha', val=0.0, units='deg')
indep_var_comp.add_output('beta', val=0.0, units='deg')
indep_var_comp.add_output('Mach_number', val=0.1)                   # 70 mph approx. TODO: make exact
indep_var_comp.add_output('re', val=1.0e6, units='1/m')
indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
indep_var_comp.add_output('CT', val=grav_constant * 17.e-6, units='1/s')
indep_var_comp.add_output('R', val=11.165e6, units='m')
indep_var_comp.add_output('W0', val=0.4 * 3e5,  units='kg')
indep_var_comp.add_output('speed_of_sound', val=295.4, units='m/s')
indep_var_comp.add_output('load_factor', val=1.)
indep_var_comp.add_output('empty_cg', val=np.array([262.614, 0.0, 115.861]), units='m')

prob.model.add_subsystem('prob_vars', indep_var_comp, promotes=['*'])

model.add_subsystem('geo', AerostructGeometries(surfaces=surfaces),
                    promotes_inputs=['wing_cord', 'vert_tail_area', 'horiz_tail_area'])

# Create the aero point group, which contains the actual aerodynamic
# analyses
aero_struct_group = AerostructPoint(surfaces=surfaces)
prob.model.add_subsystem('aero', aero_struct_group,
                         promotes_inputs=['v', 'alpha', 'beta', 'Mach_number', 're', 'rho',
                                          'CT', 'R', 'W0', 'speed_of_sound', 'load_factor',
                                          'empty_cg'])

# Mesh Connections for AeroStruct
for surface in surfaces:
    name = surface['name']

    prob.model.connect(f'geo.{name}.local_stiff_transformed', f'aero.coupled.{name}.local_stiff_transformed')
    prob.model.connect(f'geo.{name}.nodes', f'aero.coupled.{name}.nodes')

    ## Connect aerodyamic mesh to coupled group mesh
    prob.model.connect(f'geo.{name}_mesh', f'aero.coupled.{name}.mesh')

    ## Connect performance calculation variables
    for vname in ['radius', 'thickness', 'nodes']:
        prob.model.connect(f'geo.{name}.{vname}', f'aero.{name}_perf.{vname}')

    for vname in ['cg_location', 'structural_mass', ]:
        prob.model.connect(f'geo.{name}.{vname}', f'aero.total_perf.{name}_{vname}')

    prob.model.connect(f'geo.{name}:t_over_c', f'aero.{name}_perf.t_over_c')

prob.setup(force_alloc_complex=True)

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

# Solver settings
#prob.model.aero.coupled.nonlinear_solver.options['maxiter'] = 300

prob.run_model()

wrt = ['alpha', 'beta', 'wing_cord', 'vert_tail_area', 'horiz_tail_area']
of = ['aero.CL', 'aero.CD', 'aero.CM']

totals = prob.compute_totals(of=of, wrt=wrt)
print(totals)
print('done')