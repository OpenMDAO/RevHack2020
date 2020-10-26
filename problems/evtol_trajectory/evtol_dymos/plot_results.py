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
energy = sol.get_val('traj.phase0.timeseries.states:energy')

time_exp = sim.get_val('traj.phase0.timeseries.time')
power_exp = sim.get_val('traj.phase0.timeseries.controls:power')
x_exp = sim.get_val('traj.phase0.timeseries.states:x')
y_exp = sim.get_val('traj.phase0.timeseries.states:y')
energy_exp = sim.get_val('traj.phase0.timeseries.states:energy')

fix, ax = plt.subplots(1, 1)
ax.plot(time, power, 'ro')
ax.plot(time_exp, power_exp, 'b-')

fix, ax = plt.subplots(1, 1)
ax.plot(x, y, 'ro')
ax.plot(x_exp, y_exp, 'b-')

fix, ax = plt.subplots(1, 1)
ax.plot(time, energy, 'ro')
ax.plot(time_exp, energy_exp, 'b-')


plt.show()