import sys
sys.path.append("../ode") 
# since this repo is not a real python package, we have to use this approach to being able to import stuff

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

prob = om.Problem()
prob.model.add_subsystem('ode', Dynamics(input_dict=input_dict, num_nodes=1))
prob.setup()


# make some splines to define the time history 

t_final = 28.36866175

# 20 control points 
times = np.linspace(0, 1, num_cp) 
# Shamsheer's problem used a sinusoidal distribution of control points 
times = 0.5 * (1.0 + np.sin(-0.5 * np.pi + times * np.pi))
times *= t_final


thetas = [0.07188392, 0.17391331, 0.34028059, 0.5101345 , 0.66561472,
          0.76287459, 0.84858573, 0.91466015, 0.96113079, 0.99573339,
          1.00574242, 1.01472168, 1.03595189, 1.10451423, 1.16461664,
          1.22062094, 1.28553096, 1.3307819 , 1.40092757, 1.43952015]
powers = [207161.23632379, 239090.09259429, 228846.07476655, 228171.35928472,
          203168.64876755, 214967.45622033, 215557.60195517, 224144.75074625,
          234546.06852611, 248761.85655837, 264579.96329677, 238568.31766929,
          238816.66768314, 236739.41494728, 244041.61634308, 242472.86320043,
          239606.77670727, 277307.47563171, 225369.8825676 , 293871.23097611]

theta_spline = interp1d(times, thetas, kind='cubic')
power_spline = interp1d(times, powers, kind='cubic')


#################################
# function wrapper for solve_ivp
#################################   
def f(t, y): 

    # unpack the state vector from scipy
    x, y, vx, vy, energy = y 

    power = power_spline(t)
    theta = theta_spline(t)

    # the ODE only depends on these two states
    prob['ode.theta'] = theta
    prob['ode.power'] = power
    prob['ode.vx'] = vx
    prob['ode.vy'] = vy

    prob.run_model()

    dx_dt = prob['ode.x_dot'][0]
    dy_dt = prob['ode.y_dot'][0]
    dvx_dt = prob['ode.a_x'][0]
    dvy_dt = prob['ode.a_y'][0]
    denergy_dt = prob['ode.energy_dot'][0]

    return [dx_dt, dy_dt, dvx_dt, dvy_dt, denergy_dt]

y0 = [0,.01, 0., .01, 0.]

result = solve_ivp(f, (0, t_final), y0)

times = result.t
x = result.y[0,:]
y = result.y[1,:]
x_dot = result.y[2,:]
y_dot = result.y[3,:]
energy = result.y[4,:]


import matplotlib.pylab as plt

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()



