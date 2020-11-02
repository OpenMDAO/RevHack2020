# --------------------------------------------------------------------------------------------------
# This contains the OpenMDAO run script (this works with OpenMDAO version 2.6.0)
# This requires the SNOPT optimizer, but options for the free SciPy optimizer are also included.
# SI units used for everthing unless otherwise specified.
# Author: Shamsheer Chauhan
# --------------------------------------------------------------------------------------------------

from __future__ import division, print_function
import sys
import numpy as np
from openmdao.api import Problem, ScipyOptimizeDriver, IndepVarComp, ExplicitComponent, n2
from openmdao.api import pyOptSparseDriver, SqliteRecorder
from bsplines_comp import BsplinesComp
from complex_transition_components import Dynamics

# Input arguments are required when calling this script.
# The first argument is the induced-velocity factor in percentage (e.g., 0, 50, 100).
input_arg_1 = sys.argv[1]
# The second is the stall option: 's' allows stall or 'ns' does not allow stall.
input_arg_2 = sys.argv[2]

if input_arg_2 == 'ns':
    print("STALL CONSTRAINT IS ON")
elif input_arg_2 != 's':
    print("Incorrect stall option, should be ns or s"); exit()

prob = Problem()

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

indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

# INITIAL GUESSES FOR DESIGN VARIABLES
indeps.add_output('flight_time', val=20.)
indeps.add_output('powers_cp', val=np.ones(num_cp)*2e5)
indeps.add_output('thetas_cp', val=np.ones(num_cp)*np.pi/5.)

# ADD B-SPLINE COMPONENTS TO THE SYSTEM
prob.model.add_subsystem('b_spline_power', BsplinesComp(num_control_points = num_cp, num_points = num_steps, in_name = 'powers_cp', out_name = 'powers'), promotes=['*'])
prob.model.add_subsystem('b_spline_thetas', BsplinesComp(num_control_points = num_cp, num_points = num_steps, in_name = 'thetas_cp', out_name = 'thetas'), promotes=['*'])

# ADD THE MAIN PHYSICS COMPONENT TO THE SYSTEM
prob.model.add_subsystem('dynamics', Dynamics(input_dict=input_dict), promotes=['*'])

# OBJECTIVE
prob.model.add_objective('energy', scaler = 2e-7)

# DESIGN VARIABLES
prob.model.add_design_var('powers_cp', lower = 1e3, upper = 311000, scaler=5e-6)
prob.model.add_design_var('thetas_cp', lower = 0., upper = 3*np.pi/4, scaler=1.2)
prob.model.add_design_var('flight_time', lower = 5., upper = 60., scaler = 3e-2)

# CONSTRAINTS
prob.model.add_constraint('y', lower=305, scaler = 3e-3) # Constraint for the final vertical displacement
prob.model.add_constraint('x', equals=900, scaler = 3e-3) # Constraint for the final horizontal displacement
prob.model.add_constraint('y_min', lower=0.) # Constraint for the minimum vertical displacement
# prob.model.add_constraint('u_prop_min', lower=1e-6) # Constraint for the inflow velocity for the propeller
prob.model.add_constraint('x_dot', equals=67., scaler = 2e-2) # Constraint for the final horizontal speed
if input_arg_2 == 'ns': # stall constraints
    prob.model.add_constraint('aoa_max', upper=15. / 180 * np.pi, scaler = 4.)
    prob.model.add_constraint('aoa_min', lower=-15. / 180 * np.pi, scaler = 4.)
prob.model.add_constraint('acc_max', upper = 0.3, scaler = 4.) # Constraint for the acceleration magnitude

prob.model.approx_totals(method='cs', step=1e-30) # Use the complex step method for total derivatives
# prob.model.approx_totals(method='fd', form='central')

# Options for the SNOPT optimizer
prob.driver = pyOptSparseDriver()
prob.driver.add_recorder(SqliteRecorder("cases.sql"))
prob.driver.options['optimizer'] = "SNOPT"
prob.driver.opt_settings['Major optimality tolerance'] = 1e-8
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-8
prob.driver.opt_settings['Major iterations limit'] = 400

## Uncomment following lines for ScipyOptimizeDriver. Don't forget to comment out above driver options.
## Tuning the above scalers and initial guesses may be necessary.
# from openmdao.api import ScipyOptimizeDriver
# prob.driver = ScipyOptimizeDriver()
# prob.driver.add_recorder(SqliteRecorder("cases.sql"))
# prob.driver.options['tol'] = 1e-8
# prob.driver.options['maxiter'] = 200
# prob.driver.options['disp'] = True

prob.setup(check=True)

prob.run_driver()

# Save the design variables and states
np.save("y", prob.model.dynamics.y)
np.save("x", prob.model.dynamics.x)
np.save("y_dots", prob.model.dynamics.y_dots)
np.save("x_dots", prob.model.dynamics.x_dots)
np.save("thetas", prob.model.dynamics.thetas)
np.save("powers", prob.model.dynamics.powers)
np.save("atov", prob.model.dynamics.atov)
np.save("thrusts", prob.model.dynamics.thrusts)
np.save("energy", prob.model.dynamics.energy)
np.save("ft", prob.model.dynamics.flight_time)
np.save("CD", prob.model.dynamics.CD)
np.save("CL", prob.model.dynamics.CL)
np.save("aoa", prob.model.dynamics.aoa)
np.save("min_u_prop", prob.model.dynamics.min_u_prop)
np.save("acc", prob.model.dynamics.acc)
np.save("L_wings", prob.model.dynamics.L_wings)
np.save("D_wings", prob.model.dynamics.D_wings)
np.save("D_fuse", prob.model.dynamics.D_fuse)
np.save("N", prob.model.dynamics.N)
np.save("aoa_prop", prob.model.dynamics.aoa_prop)
np.save("a_x", prob.model.dynamics.a_x)
np.save("a_y", prob.model.dynamics.a_y)
