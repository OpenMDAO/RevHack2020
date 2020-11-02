import sys

sys.path.append("../ode")

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

import openmdao.api as om

from evtol_dynamics_comp import Dynamics

from bsplines_comp import BsplinesComp

# The first argument is the induced-velocity factor in percentage (e.g., 0, 50, 100).
input_arg_1 = 0
# The second is the stall option: 's' allows stall or 'ns' does not allow stall.
input_arg_2 = 'ns'

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

#######################################################
# Settings from Shamsheer's original code
#######################################################

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
              'num_steps': num_steps,  # number of time steps
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

# make some splines to define the time history

t_final = 28.36866175
# using the sine_distribution helper from the SplineComp docs
# http://openmdao.org/twodocs/versions/3.4.0/features/building_blocks/components/spline_comp.html#splinecomp-interpolation-distribution
time_cp = om.sine_distribution(num_cp, start=0, end=1, phase=np.pi)

theta_cp = [0.07188392, 0.17391331, 0.34028059, 0.5101345, 0.66561472,
            0.76287459, 0.84858573, 0.91466015, 0.96113079, 0.99573339,
            1.00574242, 1.01472168, 1.03595189, 1.10451423, 1.16461664,
            1.22062094, 1.28553096, 1.3307819, 1.40092757, 1.43952015]
power_cp = [207161.23632379, 239090.09259429, 228846.07476655, 228171.35928472,
            203168.64876755, 214967.45622033, 215557.60195517, 224144.75074625,
            234546.06852611, 248761.85655837, 264579.96329677, 238568.31766929,
            238816.66768314, 236739.41494728, 244041.61634308, 242472.86320043,
            239606.77670727, 277307.47563171, 225369.8825676, 293871.23097611]


class RK4Integration(om.ExplicitComponent):

    def setup(self):
        sub_prob = om.Problem()
        sub_prob.model.add_subsystem('time_calc', om.ExecComp('norm_time=time/t_final',
                                                              time={'units': 's'},
                                                              t_final={'units': 's'}))
        controls = sub_prob.model.add_subsystem('controls', om.MetaModelStructuredComp(method='scipy_cubic',
                                                                                       training_data_gradients=True))
        controls.add_input('norm_time', 0.0, training_data=time_cp)
        controls.add_output('theta', 0.0, training_data=theta_cp)
        controls.add_output('power', 0.0, training_data=power_cp)
        sub_prob.model.connect('time_calc.norm_time', 'controls.norm_time')

        sub_prob.model.add_subsystem('ode', Dynamics(input_dict=input_dict, num_nodes=1))
        sub_prob.model.connect('controls.theta', 'ode.theta')
        sub_prob.model.connect('controls.power', 'ode.power')

        sub_prob.setup()

        self.sub_prob = sub_prob

        self.add_input('t_final', val=30., units='s')
        self.add_input('time', val=0., units='s')
        self.add_input('theta_cp', val=np.ones(num_cp), units='rad')
        self.add_input('power_cp', val=np.ones(num_cp), units='W')

        self.add_output('x', shape=num_steps, units='m')
        self.add_output('y', shape=num_steps, units='m')
        self.add_output('vx', shape=num_steps, units='m/s')
        self.add_output('vy', shape=num_steps, units='m/s')
        self.add_output('energy', shape=num_steps, units='J')
        self.add_output('times', shape=num_steps, units='s')

    def rk4_weight(self, t, vx, vy):
        self.sub_prob['time_calc.time'] = t
        self.sub_prob['ode.vx'] = vx
        self.sub_prob['ode.vy'] = vy
        self.sub_prob.run_model()
        dx_dt = self.sub_prob['ode.x_dot'][0]
        dy_dt = self.sub_prob['ode.y_dot'][0]
        dvx_dt = self.sub_prob['ode.a_x'][0]
        dvy_dt = self.sub_prob['ode.a_y'][0]
        denergy_dt = self.sub_prob['ode.energy_dot'][0]
        f = np.array([dx_dt, dy_dt, dvx_dt, dvy_dt, denergy_dt])
        return f

    def compute(self, inputs, outputs):
        sub_prob = self.sub_prob

        outputs['x'][0] = 0
        outputs['y'][0] = 0.01
        outputs['vx'][0] = 0
        outputs['vy'][0] = .01
        outputs['energy'][0] = 0
        outputs['times'][0] = 0

        dt = t_final / num_steps

        sub_prob['time_calc.t_final'] = inputs['t_final']
        for i in range(num_steps - 1):
            sub_prob['time_calc.time'] = outputs['times'][i].copy()

            k1 = dt * self.rk4_weight(sub_prob['time_calc.time'], outputs['vx'][i], outputs['vy'][i])
            k2 = dt * self.rk4_weight(sub_prob['time_calc.time'] + 0.5 * dt, outputs['vx'][i] + 0.5 * k1[2],
                                      outputs['vy'][i] + 0.5 * k1[3])

            k3 = dt * self.rk4_weight(sub_prob['time_calc.time'] + 0.5 * dt, outputs['vx'][i] + 0.5 * k2[2],
                                      outputs['vy'][i] + 0.5 * k2[3])

            k4 = dt * self.rk4_weight(sub_prob['time_calc.time'] + dt, outputs['vx'][i] + k3[2],
                                      outputs['vy'][i] + k3[3])

            outputs['x'][i + 1] = outputs['x'][i] + k1[0]/6 + k2[0]/3 + k3[0]/3 + k4[0]/6
            outputs['y'][i + 1] = outputs['y'][i] + k1[1]/6 + k2[1]/3 + k3[1]/3 + k4[1]/6
            outputs['vx'][i + 1] = outputs['vx'][i] + k1[2]/6 + k2[2]/3 + k3[2]/3 + k4[2]/6
            outputs['vy'][i + 1] = outputs['vy'][i] + k1[3]/6 + k2[3]/3 + k3[3]/3 + k4[3]/6
            outputs['energy'][i + 1] = outputs['energy'][i] + k1[4]/6 + k2[4]/3 + k3[4]/3 + k4[4]/6

            outputs['times'][i + 1] = outputs['times'][i] + dt


if __name__ == "__main__":
    import matplotlib.pylab as plt

    p = om.Problem()
    p.model = RK4Integration()

    p.setup()

    p['theta_cp'] = theta_cp
    p['power_cp'] = power_cp

    p.run_model()

    fig, ax = plt.subplots()
    ax.plot(p['x'], p['y'])
    plt.show()
