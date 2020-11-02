import openmdao.api as om
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "../ode")

import verify_data

mpl.rcParams['lines.markersize'] = 4

sol_case = om.CaseReader('dymos_solution.db').get_case('final')
sim_case = om.CaseReader('dymos_simulation.db').get_case(-1)

sol = {}
sol['time']  = sol_case.get_val('traj.phase0.timeseries.time')
sol['power'] = sol_case.get_val('traj.phase0.timeseries.controls:power')
sol['theta'] = sol_case.get_val('traj.phase0.timeseries.controls:theta')
sol['x'] = sol_case.get_val('traj.phase0.timeseries.states:x')
sol['y'] = sol_case.get_val('traj.phase0.timeseries.states:y')
sol['vx'] = sol_case.get_val('traj.phase0.timeseries.states:vx')
sol['vy'] = sol_case.get_val('traj.phase0.timeseries.states:vy')
sol['energy'] = sol_case.get_val('traj.phase0.timeseries.states:energy')
sol['acc'] = sol_case.get_val('traj.phase0.timeseries.acc')
sol['aoa'] = sol_case.get_val('traj.phase0.timeseries.aoa')
sol['thrust'] = sol_case.get_val('traj.phase0.timeseries.thrust')
sol['cl'] = sol_case.get_val('traj.phase0.timeseries.CL')
sol['cd'] = sol_case.get_val('traj.phase0.timeseries.CD')

sim = {}
sim['time'] = sim_case.get_val('traj.phase0.timeseries.time')
sim['power'] = sim_case.get_val('traj.phase0.timeseries.controls:power')
sim['theta'] = sim_case.get_val('traj.phase0.timeseries.controls:theta')
sim['x'] = sim_case.get_val('traj.phase0.timeseries.states:x')
sim['y'] = sim_case.get_val('traj.phase0.timeseries.states:y')
sim['vx'] = sim_case.get_val('traj.phase0.timeseries.states:vx')
sim['vy'] = sim_case.get_val('traj.phase0.timeseries.states:vy')
sim['energy'] = sim_case.get_val('traj.phase0.timeseries.states:energy')
sim['acc'] = sim_case.get_val('traj.phase0.timeseries.acc')
sim['aoa'] = sim_case.get_val('traj.phase0.timeseries.aoa')
sim['thrust'] = sim_case.get_val('traj.phase0.timeseries.thrust')
sim['cl'] = sim_case.get_val('traj.phase0.timeseries.CL')
sim['cd'] = sim_case.get_val('traj.phase0.timeseries.CD')

verif = {}
verif['theta'] = np.load('../original/thetas.npy')
verif['power'] = np.load('../original/powers.npy')
steps = verify_data.thetas.size
verif['time'] = np.linspace(0, verify_data.flight_time, steps)
verif['x'] = np.load('../original/x.npy')
verif['y'] = np.load('../original/y.npy')
verif['vx'] = np.load('../original/x_dots.npy')
verif['vy'] = np.load('../original/y_dots.npy')
verif['energy'] = np.load('../original/energy.npy')[:-1]
verif['ax'] = np.load('../original/a_x.npy')
verif['ay'] = np.load('../original/a_y.npy')
verif['acc'] = np.load('../original/acc.npy')
verif['aoa'] = np.load('../original/aoa.npy')
verif['thrust'] = verify_data.thrusts[:-1]
verif['cl'] = verify_data.CL[:-1]
verif['cd'] = verify_data.CD[:-1]

print(sol['energy'])
print(sim['energy'])
print(verif['energy'])

def plot_on_axes(x, y, axes, upper=None, lower=None):
    axes.plot(sol[x], sol[y].ravel(), 'o')
    axes.plot(sim[x], sim[y].ravel(), '-')
    axes.plot(verif[x], verif[y].ravel(), 'k:')
    if upper is not None:
        axes.plot([0, 30], [upper, upper], 'r--')
    if lower is not None:
        axes.plot([0, 30], [lower, lower], 'r--')

    axes.set_xlabel(x)
    axes.set_ylabel(y)

plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(3, 3, figsize=(15,6))

axes = np.reshape(axes, (9,))

plot_on_axes('time', 'power', axes[0])
plot_on_axes('time', 'theta', axes[1])
plot_on_axes('x', 'y', axes[2])
plot_on_axes('time', 'energy', axes[3])
plot_on_axes('time', 'aoa', axes[4], upper=np.radians(15))
plot_on_axes('time', 'thrust', axes[5])
plot_on_axes('time', 'acc', axes[6], upper=0.3)
plot_on_axes('time', 'cl', axes[7])
plot_on_axes('time', 'cd', axes[8])

plt.tight_layout()

plt.savefig('results.png')

plt.show()