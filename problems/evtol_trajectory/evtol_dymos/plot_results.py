import openmdao.api as om
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../ode")

import verify_data


sol = om.CaseReader('solution.sql').get_case('final')
sim = om.CaseReader('simulation.sql').get_case(-1)

time = sol.get_val('traj.phase0.timeseries.time')
power = sol.get_val('traj.phase0.timeseries.controls:power')
theta = sol.get_val('traj.phase0.timeseries.controls:theta')
x = sol.get_val('traj.phase0.timeseries.states:x')
y = sol.get_val('traj.phase0.timeseries.states:y')
vx = sol.get_val('traj.phase0.timeseries.states:vx')
vy = sol.get_val('traj.phase0.timeseries.states:vy')
energy = sol.get_val('traj.phase0.timeseries.states:energy')
acc = sol.get_val('traj.phase0.timeseries.acc')

time_exp = sim.get_val('traj.phase0.timeseries.time')
power_exp = sim.get_val('traj.phase0.timeseries.controls:power')
theta_exp = sim.get_val('traj.phase0.timeseries.controls:theta')
x_exp = sim.get_val('traj.phase0.timeseries.states:x')
y_exp = sim.get_val('traj.phase0.timeseries.states:y')
vx_exp = sim.get_val('traj.phase0.timeseries.states:vx')
vy_exp = sim.get_val('traj.phase0.timeseries.states:vy')
energy_exp = sim.get_val('traj.phase0.timeseries.states:energy')
acc_exp = sim.get_val('traj.phase0.timeseries.acc')

steps = verify_data.thetas.size
time_verif = np.linspace(0, verify_data.flight_time, steps)
theta_verif = verify_data.thetas
power_verif = verify_data.powers
x_verif = verify_data.x
y_verif = verify_data.y
vx_verif = verify_data.x_dot
vy_verif = verify_data.y_dot
ax_verif = verify_data.a_x
ay_verif = verify_data.a_y
acc_verif = np.sqrt(ax_verif**2 + ay_verif**2) / 9.81
power_verif = verify_data.powers


fix, axes = plt.subplots(2, 3)
axes[0, 0].plot(time, power, 'o')
axes[0,0].plot(time_exp, power_exp, '-')
import numpy as np
axes[0,0].plot(time_verif, power_verif.ravel(), 'k:')

# fix, ax = plt.subplots(1, 1)
axes[0,1].plot(x, y, 'o')
axes[0,1].plot(x_exp, y_exp, '-')
axes[0,1].plot(x_verif, y_verif, 'k:')
axes[0,1].set_xlabel('x')
axes[0,1].set_ylabel('y')

# fix, ax = plt.subplots(1, 1)
axes[0,2].plot(time, energy, 'o')
axes[0,2].plot(time_exp, energy_exp, '-')
axes[0,2].plot(time_verif, 6749880.06906939 * np.ones_like(time_verif), 'k:')
print(energy[-1])
axes[0,2].set_xlabel('time')
axes[0,2].set_ylabel('energy')

# fix, ax = plt.subplots(1, 1)
axes[1,0].plot(time, acc, 'o')
axes[1,0].plot(time_exp, acc_exp, '-')
axes[1,0].plot(time_verif, acc_verif, 'k:')
axes[1,0].set_xlabel('time')
axes[1,0].set_ylabel('acc')
axes[1,0].set_ylim(0, 1)

# fix, ax = plt.subplots(1, 1)
axes[1,1].plot(time, vx, 'o')
axes[1,1].plot(time, vy, 'o')
axes[1,1].plot(time_exp, vx_exp, '-')
axes[1,1].plot(time_exp, vy_exp, '-')
axes[1,1].plot(time_verif, vx_verif[:-1], 'k:')
axes[1,1].plot(time_verif, vy_verif[:-1], 'k:')
axes[1,1].set_xlabel('time')
axes[1,1].set_ylabel('vx and vy')

axes[1,2].plot(time, theta, 'o')
axes[1,2].plot(time_exp, theta_exp, '-')
axes[1,2].plot(time_verif, theta_verif.ravel(), 'k:')
axes[1,2].set_xlabel('time')
axes[1,2].set_ylabel('theta')



plt.show()