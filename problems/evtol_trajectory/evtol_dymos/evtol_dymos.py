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

    phase = dm.Phase(transcription=dm.Radau(num_segments=5, order=3, solve_segments=True, compressed=False),
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
    phase.add_objective('energy', loc='final', ref=1E6)

    # Boundary Constraints
    phase.add_boundary_constraint('y', loc='final', lower=305, scaler=3e-3)  # Constraint for the final vertical displacement
    phase.add_boundary_constraint('x', loc='final', equals=900, scaler=3e-3)  # Constraint for the final horizontal displacement
    phase.add_boundary_constraint('x_dot', loc='final', equals=67., scaler=2e-2)  # Constraint for the final horizontal speed
    
    # Path Constraints
    phase.add_path_constraint('y', lower=0., upper=305, ref=300)  # Constraint for the minimum vertical displacement
    phase.add_path_constraint('acc', upper=0.3, scaler=4.)  # Constraint for the acceleration magnitude
    phase.add_path_constraint('aoa', lower=-15, upper=15, ref0=-10, ref=10)  # Constraint for the acceleration magnitude
    phase.add_path_constraint('thrust', lower=10, ref=10)  # Constraint for the thrust magnitude

    # Setup the driver
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = "SNOPT"
    p.driver.opt_settings['Major optimality tolerance'] = 1e-7
    p.driver.opt_settings['Major feasibility tolerance'] = 1e-7
    p.driver.opt_settings['Major iterations limit'] = 1000
    p.driver.opt_settings['Minor iterations limit'] = 100000
    # p.driver.opt_settings['Verify level'] = 3
    p.driver.opt_settings['iSumm'] = 6

    # p.driver.options['optimizer'] = "IPOPT"
    # p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    # p.driver.opt_settings['print_level'] = 5

    p.driver.declare_coloring(tol=1.0E-8)

    p.add_recorder(om.SqliteRecorder("solution.sql"))

    p.setup()

    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 30.0)
    p.set_val('traj.phase0.states:x', phase.interpolate(ys=[0, 900], nodes='state_input'))
    p.set_val('traj.phase0.states:y',  phase.interpolate(ys=[0, 300], nodes='state_input'))
    p.set_val('traj.phase0.states:vx', phase.interpolate(ys=[0, 20], nodes='state_input'))
    p.set_val('traj.phase0.states:vy', phase.interpolate(ys=[0.01, 0.01], nodes='state_input'))
    p.set_val('traj.phase0.states:energy', phase.interpolate(ys=[0, 1E6], nodes='state_input'))
    p.set_val('traj.phase0.controls:power', 200000.0)
    p.set_val('traj.phase0.controls:theta', phase.interpolate(ys=[0.05, np.radians(80)], nodes='control_input'))

    p.set_val('traj.phase0.controls:power', phase.interpolate(xs=np.linspace(0, 28.368, 20),
                                                              ys=[207161.23632379, 239090.09259429, 228846.07476655, 228171.35928472,
       203168.64876755, 214967.45622033, 215557.60195517, 224144.75074625,
       234546.06852611, 248761.85655837, 264579.96329677, 238568.31766929,
       238816.66768314, 236739.41494728, 244041.61634308, 242472.86320043,
       239606.77670727, 277307.47563171, 225369.8825676 , 293871.23097611], nodes='control_input'))
    p.set_val('traj.phase0.controls:theta', phase.interpolate(xs=np.linspace(0, 28.368, 20),
                                                              ys=[0.07188392, 0.17391331, 0.34028059, 0.5101345 , 0.66561472,
       0.76287459, 0.84858573, 0.91466015, 0.96113079, 0.99573339,
       1.00574242, 1.01472168, 1.03595189, 1.10451423, 1.16461664,
       1.22062094, 1.28553096, 1.3307819 , 1.40092757, 1.43952015], nodes='control_input'))


    dm.run_problem(p, refine_iteration_limit=0)
    # p.run_model()
    # with np.printoptions(linewidth=1024):
    #     p.check_totals(compact_print=True)

    p.record('final')

    exp_out = traj.simulate(record_file='simulation.sql')

    import matplotlib.pyplot as plt
    plt.plot(exp_out.get_val('traj.phase0.timeseries.states:x'),
             exp_out.get_val('traj.phase0.timeseries.states:y'))
    plt.show()
