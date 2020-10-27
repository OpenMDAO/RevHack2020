"""
Component wrapper that contains an OpenMDAO problem with VSP and OpenAeroStruct.

Executing the component performs analysis on the ecrm model and computation of total derivatives
which are returned as component outputs.
"""
import numpy as np

import openmdao.api as om

from openaerostruct.integration.aerostruct_groups import AerostructPoint
from openaerostruct.utils.constants import grav_constant

from aerostruct_vsp_groups import AerostructGeometries
from vsp_eCRM import VSPeCRM


class ECRM(om.ExplicitComponent):
    """
    Component wrapper that contains an OpenMDAO problem with VSP and OpenAeroStruct.

    Attributes
    ----------
    _problem : <Problem>
        OpenMDAO Problem that this component runs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._problem = None

    def initialize(self):
        self.options.declare('wing_surface', types=dict, default={},
                             desc="Dict containing settings that define the OAS surface "
                             "for the wing.")
        self.options.declare('horiz_tail_surface', types=dict, default={},
                             desc="Dict containing settings that define the OAS surface "
                             "for the horizontal tail.")
        self.options.declare('vert_tail_surface', types=dict, default={},
                             desc="Dict containing settings that define the OAS surface "
                             "for the vertical tail.")

    def setup(self):

        # General Inputs
        self.add_input('v', val=248.136, units='m/s')
        self.add_input('alpha', val=0.0, units='deg')
        self.add_input('beta', val=0.0, units='deg')
        self.add_input('Mach_number', val=0.1)
        self.add_input('re', val=1.0e6, units='1/m')
        self.add_input('rho', val=0.38, units='kg/m**3')
        self.add_input('CT', val=grav_constant * 17.e-6, units='1/s')
        self.add_input('R', val=11.165e6, units='m')
        self.add_input('W0', val=0.4 * 3e5,  units='kg')
        self.add_input('speed_of_sound', val=295.4, units='m/s')
        self.add_input('load_factor', val=1.)
        self.add_input('empty_cg', val=np.zeros((3)), units='m')

        # VSP Geometry Inputs
        # TODO - Original model gave no units. Probably inches, but check.
        self.add_input('wing_cord', val=59.05128)
        self.add_input('vert_tail_area', val=2295.)
        self.add_input('horiz_tail_area', val=6336.)

        # Outputs
        self.add_output('CL', 0.0)
        self.add_output('CD', 0.0)

        self.add_output('CM_alpha', 0.0)
        self.add_output('CL_alpha', 0.0)
        self.add_output('CN_beta', 0.0)

        self.add_output('C_weight', 0.0)
        self.add_output('S', 0.0)

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        prob = self._problem

        # Build the model the first time we run.
        if prob is None:
            opt = self.options
            wing_surface = opt['wing_surface']
            horiz_tail_surface = opt['horiz_tail_surface']
            vert_tail_surface = opt['vert_tail_surface']
            surfaces = [wing_surface, horiz_tail_surface, vert_tail_surface]

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

            prob.setup()
            self._problem = prob

        # Set new values.
        prob.set_val('v', inputs['v'])
        prob.set_val('alpha', inputs['alpha'])
        prob.set_val('beta', inputs['beta'])
        prob.set_val('Mach_number', inputs['Mach_number'])
        prob.set_val('re', inputs['re'])
        prob.set_val('rho', inputs['rho'])
        prob.set_val('CT', inputs['CT'])
        prob.set_val('R', inputs['R'])
        prob.set_val('W0', inputs['W0'])
        prob.set_val('speed_of_sound', inputs['speed_of_sound'])
        prob.set_val('load_factor', inputs['load_factor'])
        prob.set_val('empty_cg', inputs['empty_cg'])

        prob.set_val('wing_cord', inputs['wing_cord'])
        prob.set_val('vert_tail_area', inputs['vert_tail_area'])
        prob.set_val('horiz_tail_area', inputs['horiz_tail_area'])

        # Run Problem
        prob.run_model()

        # Extract Outputs
        outputs['CL'] = prob.get_val('aero.CL')
        outputs['CD'] = prob.get_val('aero.CD')

        rho = inputs['rho']
        v = inputs['v']
        S = prob.get_val('aero.total_perf.S_ref_total')
        W = prob.get_val('aero.L_equals_W')
        outputs['C_weight'] = W / (0.5 * rho * v**2 * S)

        # Compute Stability Derivatives
        of = ['aero.CL', 'aero.CD', 'aero.CM']
        wrt = ['alpha', 'beta']
        totals = prob.compute_totals(of=of, wrt=wrt)

        # Extract Stability Derivatives
        outputs['CM_alpha'] = totals['aero.CM', 'alpha'][1, 0]
        outputs['CL_alpha'] = totals['aero.CL', 'alpha'][0, 0]
        outputs['CN_beta'] = totals['aero.CM', 'beta'][2, 0]


if __name__ == "__main__":
    import pickle

    #Read baseline mesh
    with open('baseline_meshes_reduced.pkl', "rb") as f:
        meshes = pickle.load(f)

    wing_mesh = meshes['wing_mesh']
    horiz_tail_mesh = meshes['horiz_tail_mesh']
    vert_tail_mesh = meshes['vert_tail_mesh']

    # Define data for surfaces
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

        'thickness_cp' : np.array([.1, .2]),

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
        'mrho' : 3.e3,          # [kg/m^3] material density
        'fem_origin' : 0.35,    # normalized chordwise location of the spar
        'wing_weight_ratio' : 2.,
        'struct_weight_relief' : False,    # True to add the weight of the structure to the loads on the structure
        'distributed_fuel_weight' : False,
        # Constraints
        'exact_failure_constraint' : False, # if false, use KS function
    }

    prob = om.Problem()
    model = prob.model

    model.add_subsystem('ecrm', ECRM(wing_surface=wing_surface,
                                     horiz_tail_surface=horiz_tail_surface,
                                     vert_tail_surface=vert_tail_surface),
                        promotes=['*'])

    prob.setup()

    prob.run_model()

    print('stability derivs')
    print('CM_alpha', prob.get_val('CM_alpha'))
    print('CL_alpha', prob.get_val('CL_alpha'))
    print('CN_beta', prob.get_val('CN_beta'))

    print('done')