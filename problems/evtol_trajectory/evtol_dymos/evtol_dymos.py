import openmdao.api as om
import dymos as dm
import numpy as np

import sys
sys.path.insert(0, "../ode")

from evtol_dynamics_comp import Dynamics


if __name__ == '__main__':

    input_arg_1 = 0.0

    input_arg_2 = 'ns'

    # Some specifications
    prop_rad = 0.75
    wing_S = 9.
    wing_span = 6.
    num_blades = 3.
    blade_chord = 0.1
    num_props = 8

    # User-specified input dictionary
    input_dict = {'T_guess': 9.8 * 725 * 1.2,  # initial thrust guess
                  'x_dot_initial': 0.,  # initial horizontal speed
                  'y_dot_initial': 0.01,  # initial vertical speed
                  'y_initial': 0.01,  # initial vertical displacement
                  'A_disk': np.pi * prop_rad ** 2 * num_props,  # total propeller disk area
                  'AR': wing_span ** 2 / (0.5 * wing_S),  # aspect ratio of each wing
                  'e': 0.68,  # span efficiency factor of each wing
                  't_over_c': 0.12,  # airfoil thickness-to-chord ratio
                  'S': wing_S,  # total wing reference area
                  'CD0': 0.35 / wing_S,  # coefficient of drag of the fuselage, gear, etc.
                  'm': 725.,  # mass of aircraft
                  'a0': 5.9,  # airfoil lift-curve slope
                  'alpha_stall': 15. / 180. * np.pi,  # wing stall angle
                  'rho': 1.225,  # air density
                  'induced_velocity_factor': int(input_arg_1) / 100.,  # induced-velocity factor
                  'stall_option': input_arg_2,  # stall option: 's' allows stall, 'ns' does not
                  'R': prop_rad,  # propeller radius
                  'solidity': num_blades * blade_chord / np.pi / prop_rad,  # solidity
                  'omega': 136. / prop_rad,  # angular rotation rate
                  'prop_CD0': 0.012,  # CD0 for prop profile power
                  'k_elec': 0.9,  # electrical and mechanical losses factor
                  'k_ind': 1.2,  # induced-losses factor
                  'nB': num_blades,  # number of blades per propeller
                  'bc': blade_chord,  # representative blade chord
                  'n_props': num_props  # number of propellers
                  }

    p = om.Problem()

    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)

    phase = dm.Phase(transcription=dm.GaussLobatto(num_segments=5, solve_segments=False),
                     ode_class=Dynamics,
                     ode_init_kwargs={'input_dict': input_dict})

    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(5, 60), duration_scaler=3e-2)
    phase.add_state('x', fix_initial=True, rate_source='x_dot', ref=900, defect_ref=100)
    phase.add_state('y', fix_initial=True, rate_source='y_dot')
    phase.add_state('vx', fix_initial=True, rate_source='vx_dot')
    phase.add_state('vy', fix_initial=True, rate_source='vy_dot')
    phase.add_state('energy', fix_initial=True, rate_source='energy_dot', ref=1E7, defect_ref=1E5)

    phase.add_control('power', lower = 1e3, upper =311000, scaler=5e-6)
    phase.add_control('theta', lower = 0., upper =3*np.pi/4, scaler=1.2)

    # Objective
    phase.add_objective('energy', loc='final', scaler=2e-7)

    # Boundary Constraints
    phase.add_boundary_constraint('y', loc='final', lower=305, scaler=3e-3)  # Constraint for the final vertical displacement
    phase.add_boundary_constraint('x', loc='final', equals=900, scaler=3e-3)  # Constraint for the final horizontal displacement
    phase.add_boundary_constraint('x_dot', loc='final', equals=67., scaler=2e-2)  # Constraint for the final horizontal speed
    
    # Path Constraints
    phase.add_path_constraint('y', lower=0., upper=305, ref=300)  # Constraint for the minimum vertical displacement
    phase.add_path_constraint('acc', upper=0.3, scaler=4.)  # Constraint for the acceleration magnitude
    phase.add_path_constraint('aoa', lower=-10, upper=10, ref0=-10, ref=10)  # Constraint for the acceleration magnitude

    # Setup the driver
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = "SNOPT"
    p.driver.opt_settings['Major optimality tolerance'] = 1e-7
    p.driver.opt_settings['Major feasibility tolerance'] = 1e-7
    p.driver.opt_settings['Major iterations limit'] = 1000
    p.driver.opt_settings['Minor iterations limit'] = 100000
    p.driver.opt_settings['iSumm'] = 6

    # p.driver.declare_coloring(tol=1.0E-1)

    p.add_recorder(om.SqliteRecorder("solution.sql"))

    p.setup()

    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 10.0)
    p.set_val('traj.phase0.states:x', phase.interpolate(ys=[0, 900], nodes='state_input'))
    p.set_val('traj.phase0.states:y',  phase.interpolate(ys=[0, 300], nodes='state_input'))
    p.set_val('traj.phase0.states:vx', phase.interpolate(ys=[0, 20], nodes='state_input'))
    p.set_val('traj.phase0.states:vy', phase.interpolate(ys=[0.01, 0.01], nodes='state_input'))
    p.set_val('traj.phase0.states:energy', 0.0)
    p.set_val('traj.phase0.controls:power', 200000.0)
    p.set_val('traj.phase0.controls:theta', 0.05)


    p.run_driver()

    p.record('final')

    exp_out = traj.simulate(record_file='simulation.sql')

    import matplotlib.pyplot as plt
    plt.plot(exp_out.get_val('traj.phase0.timeseries.states:x'),
             exp_out.get_val('traj.phase0.timeseries.states:y'))
    plt.show()
