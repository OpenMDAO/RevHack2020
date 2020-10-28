import openmdao.api as om
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../ode")

sol = om.CaseReader('solution.sql').get_case('final')
sim = om.CaseReader('simulation.sql').get_case(-1)

time = sol.get_val('traj.phase0.timeseries.time')
power = sol.get_val('traj.phase0.timeseries.controls:power')
x = sol.get_val('traj.phase0.timeseries.states:x')
y = sol.get_val('traj.phase0.timeseries.states:y')
vx = sol.get_val('traj.phase0.timeseries.states:vx')
vy = sol.get_val('traj.phase0.timeseries.states:vy')
energy = sol.get_val('traj.phase0.timeseries.states:energy')
acc = sol.get_val('traj.phase0.timeseries.acc')

time_exp = sim.get_val('traj.phase0.timeseries.time')
power_exp = sim.get_val('traj.phase0.timeseries.controls:power')
x_exp = sim.get_val('traj.phase0.timeseries.states:x')
y_exp = sim.get_val('traj.phase0.timeseries.states:y')
vx_exp = sim.get_val('traj.phase0.timeseries.states:vx')
vy_exp = sim.get_val('traj.phase0.timeseries.states:vy')
energy_exp = sim.get_val('traj.phase0.timeseries.states:energy')
acc_exp = sim.get_val('traj.phase0.timeseries.acc')

fix, ax = plt.subplots(1, 1)
ax.plot(time, power, 'o')
ax.plot(time_exp, power_exp, '-')
import numpy as np
ax.plot(np.linspace(0, 28.368, 20),
         [207161.23632379, 239090.09259429, 228846.07476655, 228171.35928472,
          203168.64876755, 214967.45622033, 215557.60195517, 224144.75074625,
          234546.06852611, 248761.85655837, 264579.96329677, 238568.31766929,
          238816.66768314, 236739.41494728, 244041.61634308, 242472.86320043,
          239606.77670727, 277307.47563171, 225369.8825676 , 293871.23097611], 'ko')

fix, ax = plt.subplots(1, 1)
ax.plot(x, y, 'o')
ax.plot(x_exp, y_exp, '-')

fix, ax = plt.subplots(1, 1)
ax.plot(time, energy, 'o')
ax.plot(time_exp, energy_exp, '-')

fix, ax = plt.subplots(1, 1)
ax.plot(time, acc, 'o')
ax.plot(time_exp, acc_exp, '-')

fix, ax = plt.subplots(1, 1)
ax.plot(time, vx, 'o')
ax.plot(time, vy, 'o')
ax.plot(time_exp, vx_exp, '-')
ax.plot(time_exp, vy_exp, '-')


plt.show()