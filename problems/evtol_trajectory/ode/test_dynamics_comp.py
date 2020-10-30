import unittest 

import numpy as np 

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from evtol_dynamics_comp import Dynamics
from evtol_dynamics_comp_vectorized import Dynamics as DynamicsVectorized

import verify_data

class VerifyTest(unittest.TestCase): 

    def make_problem(self, comp_class=Dynamics):

        # Input arguments are required when calling this script.
        # The first argument is the induced-velocity factor in percentage (e.g., 0, 50, 100).
        input_arg_1 = 0
        # The second is the stall option: 's' allows stall or 'ns' does not allow stall.
        input_arg_2 = 'ns'


        prob = om.Problem()

        # User-specified number of B-spline control points
        num_cp = 20
        # User-specified numer of time steps
        num_steps = 500

        # Some specifications
        prop_rad = 0.75
        wing_S = 9.
        wing_span = 6.
        num_blades = 3.
        blade_chord = 0.1
        num_props = 8

        # User-specified input dictionary
        input_dict = { 'T_guess' : 9.8*725*1.2, # initial thrust guess
                        'x_dot_initial' : 0.,   # initial horizontal speed
                        'y_dot_initial' : 0.01, # initial vertical speed
                        'y_initial' : 0.01,     # initial vertical displacement
                        'A_disk' : np.pi * prop_rad**2 * num_props,    # total propeller disk area
                        'AR' : wing_span**2 / (0.5 * wing_S),  # aspect ratio of each wing
                        'e' : 0.68,             # span efficiency factor of each wing
                        't_over_c' : 0.12,      # airfoil thickness-to-chord ratio
                        'S' : wing_S,           # total wing reference area
                        'CD0' : 0.35 / wing_S,  # coefficient of drag of the fuselage, gear, etc.
                        'm' : 725.,             # mass of aircraft
                        'a0' : 5.9,             # airfoil lift-curve slope
                        'alpha_stall' : 15. / 180. * np.pi, # wing stall angle
                        'rho' : 1.225,          # air density
                        'induced_velocity_factor': int(input_arg_1)/100., # induced-velocity factor
                        'stall_option' : input_arg_2, # stall option: 's' allows stall, 'ns' does not
                        'num_steps' : num_steps, # number of time steps
                        'R' : prop_rad,         # propeller radius
                        'solidity' : num_blades * blade_chord / np.pi / prop_rad, # solidity
                        'omega' : 136. / prop_rad,  # angular rotation rate
                        'prop_CD0' : 0.012,     # CD0 for prop profile power
                        'k_elec' : 0.9,         # electrical and mechanical losses factor
                        'k_ind' : 1.2,          # induced-losses factor
                        'nB' : num_blades,      # number of blades per propeller
                        'bc' : blade_chord,     # representative blade chord
                        'n_props' : num_props   # number of propellers
                        }

        # ADD THE MAIN PHYSICS COMPONENT TO THE SYSTEM
        prob.model.add_subsystem('dynamics', comp_class(input_dict=input_dict, num_nodes=num_steps), promotes=['*'])

        prob.setup(check=True)

        return prob

    def assert_results(self, prob):
        prob['power'] = verify_data.powers
        prob['theta'] = verify_data.thetas

        prob['vx'] = verify_data.x_dot[:-1]
        prob['vy'] = verify_data.y_dot[:-1]

        prob.run_model()

        tol = 1e-6

        assert_near_equal(prob['x_dot'], verify_data.x_dot[:-1], tol)
        assert_near_equal(prob['y_dot'], verify_data.y_dot[:-1], tol)

        assert_near_equal(prob['thrust'], verify_data.thrusts[:-1], tol)
        assert_near_equal(prob['atov'], verify_data.atov[1:], tol)

        assert_near_equal(prob['CL'], verify_data.CL[1:], tol)
        assert_near_equal(prob['CD'], verify_data.CD[1:], tol)

        assert_near_equal(prob['L_wings'], verify_data.L_wings, tol)
        assert_near_equal(prob['D_wings'], verify_data.D_wings, tol)

        assert_near_equal(prob['a_y'], verify_data.a_y, tol)
        assert_near_equal(prob['a_x'], verify_data.a_x, tol)

    def test_dynamics(self):

        prob = self.make_problem(comp_class=Dynamics)

        self.assert_results(prob)

    def test_dynamics_vectorized(self):

        prob = self.make_problem(comp_class=DynamicsVectorized)

        self.assert_results(prob)





if __name__ == "__main__": 
    unittest.main()