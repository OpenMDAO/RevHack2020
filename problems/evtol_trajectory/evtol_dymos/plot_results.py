import openmdao.api as om
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../ode")

import verify_data

mpl.rcParams['lines.markersize'] = 4

sol = om.CaseReader('dymos_solution.db').get_case('final')
sim = om.CaseReader('dymos_simulation.db').get_case(-1)

time = sol.get_val('traj.phase0.timeseries.time')
power = sol.get_val('traj.phase0.timeseries.controls:power')
theta = sol.get_val('traj.phase0.timeseries.controls:theta')
x = sol.get_val('traj.phase0.timeseries.states:x')
y = sol.get_val('traj.phase0.timeseries.states:y')
vx = sol.get_val('traj.phase0.timeseries.states:vx')
vy = sol.get_val('traj.phase0.timeseries.states:vy')
energy = sol.get_val('traj.phase0.timeseries.states:energy')
acc = sol.get_val('traj.phase0.timeseries.acc')
aoa = sol.get_val('traj.phase0.timeseries.aoa')
thrust = sol.get_val('traj.phase0.timeseries.thrust')
cl = sol.get_val('traj.phase0.timeseries.CL')
cd = sol.get_val('traj.phase0.timeseries.CD')

time_exp = sim.get_val('traj.phase0.timeseries.time')
power_exp = sim.get_val('traj.phase0.timeseries.controls:power')
theta_exp = sim.get_val('traj.phase0.timeseries.controls:theta')
x_exp = sim.get_val('traj.phase0.timeseries.states:x')
y_exp = sim.get_val('traj.phase0.timeseries.states:y')
vx_exp = sim.get_val('traj.phase0.timeseries.states:vx')
vy_exp = sim.get_val('traj.phase0.timeseries.states:vy')
energy_exp = sim.get_val('traj.phase0.timeseries.states:energy')
acc_exp = sim.get_val('traj.phase0.timeseries.acc')
aoa_exp = sim.get_val('traj.phase0.timeseries.aoa')
thrust_exp = sim.get_val('traj.phase0.timeseries.thrust')
cl_exp = sim.get_val('traj.phase0.timeseries.CL')
cd_exp = sim.get_val('traj.phase0.timeseries.CD')

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
thrust_verif = verify_data.thrusts
cl_verif = verify_data.CL
cd_verif = verify_data.CD


fig, axes = plt.subplots(3, 3, figsize=(11,6))
axes[0, 0].plot(time, power, 'o')
axes[0, 0].plot(time_exp, power_exp, '-')
import numpy as np
axes[0, 0].plot(time_verif, power_verif.ravel(), 'k:')
axes[0, 0].set_xlabel('time')
axes[0, 0].set_ylabel('power')

# fix, ax = plt.subplots(1, 1)
axes[0, 1].plot(x, y, 'o')
axes[0, 1].plot(x_exp, y_exp, '-')
axes[0, 1].plot(x_verif, y_verif, 'k:')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')

# fix, ax = plt.subplots(1, 1)
axes[0, 2].plot(time, energy, 'o')
axes[0, 2].plot(time_exp, energy_exp, '-')
axes[0, 2].plot(time_verif, 6749880.06906939 * np.ones_like(time_verif), 'k:')
axes[0, 2].set_xlabel('time')
axes[0, 2].set_ylabel('energy')

# fix, ax = plt.subplots(1, 1)
axes[1, 0].plot(time, acc, 'o')
axes[1, 0].plot(time_exp, acc_exp, '-')
axes[1, 0].plot(time_verif, acc_verif, 'k:')
axes[1, 0].set_xlabel('time')
axes[1, 0].set_ylabel('acc')

# fix, ax = plt.subplots(1, 1)
axes[1, 1].plot(time, vx, 'o')
axes[1, 1].plot(time, vy, 'o')
axes[1, 1].plot(time_exp, vx_exp, '-', label='vx')
axes[1, 1].plot(time_exp, vy_exp, '-', label='vy')
axes[1, 1].plot(time_verif, vx_verif[:-1], 'k:')
axes[1, 1].plot(time_verif, vy_verif[:-1], 'k:')
axes[1, 1].set_xlabel('time')
axes[1, 1].set_ylabel('vx and vy')
axes[1, 1].legend()

axes[1, 2].plot(time, np.degrees(theta), 'o')
axes[1, 2].plot(time_exp, np.degrees(theta_exp), '-')
axes[1, 2].plot(time_verif, np.degrees(theta_verif).ravel(), 'k:')
axes[1, 2].set_xlabel('time')
axes[1, 2].set_ylabel('theta')


axes[2, 0].plot(time, np.degrees(aoa), 'o')
axes[2, 0].plot(time_exp, np.degrees(aoa_exp), '-')
# axes[2, 0].plot(time_verif, aoa_verif.ravel(), 'k:')
axes[2, 0].set_xlabel('time')
axes[2, 0].set_ylabel('aoa')

axes[2, 1].plot(time, thrust, 'o')
axes[2, 1].plot(time_exp, thrust_exp, '-')
axes[2, 1].plot(time_verif, thrust_verif[:-1], 'k:')
axes[2, 1].set_xlabel('time')
axes[2, 1].set_ylabel('thrust')

axes[2, 2].plot(time, cl, 'o')
axes[2, 2].plot(time, cd, 'o')
axes[2, 2].plot(time_exp, cl_exp, '-', label='CL')
axes[2, 2].plot(time_exp, cd_exp, '-', label='CD')
axes[2, 2].plot(time_verif, cl_verif[:-1], 'k:')
axes[2, 2].plot(time_verif, cd_verif[:-1], 'k:')
axes[2, 2].set_xlabel('time')
axes[2, 2].set_ylabel('CL ad CD')
axes[2, 2].legend()

for ax in axes.ravel():
    ax.grid()

plt.tight_layout()

plt.show()