# Conversion of the original ODE to a Dymos-Compatible ODE

In the original request, the user wants to know how to implement their problem in Dymos.
The problem, as posed, uses a simple Euler-step integration to propagate the dynamics from one time-step to the next.

## Composing the Ordinary Differential Equations (ODE)

The first step in the translation is to take the integration behavior out of the user's component and treat it as an ODE.
That is, the component's job will be to compute the derivatives of our state variables, which we will then integrate (using Dymos, a for-loop, or an external ODE-integrator).
So first order of business, let's remove the time-marching behavior.
We'll focus on the Dynamics component first.

* We're removing anything that deals explicitly with time integration (the time duration of the simulation)
* The ODE itself will not govern the maximum-minimum quantities seen in the trajectory, so those are removed.
* We're not going to specify the type as complex for our inputs and outputs, OpenMDAO can handle that for us when performing complex-step.
* The outputs we're adding to the ODE are the rates of our 5 states (x_dot, y_dot, vx_dot, vy_dot, energy_dot)
* We need to treat the values of the states at any given point as inputs. Here the ODE only depends the states `vx` and `vy`.
* The quantities which we want to track over the integration (AoA, u_prop) will be added as additional outputs of the ODE.
* Note the increment of the velocity states is taken from the `change` function in the original implementation.

```diff
import numpy as np
from openmdao.api import ExplicitComponent


class Dynamics(ExplicitComponent):
    """
    This is the OpenMDAO component that takes the design variables and computes
    the objective function and other quantities of interest.

    Parameters
    ----------
    powers : array
        Electrical power distribution as a function of time
    thetas : array
        Wing-angle-to-vertical distribution as a function of time
    flight_time : float
        Duration of flight

    Returns
    -------
    x_dot : float
        Final horizontal speed
    y_dot : float
        Final vertical speed
    x : float
        Final horizontal position
    y : float
        Final vertical position
    y_min : float
        Minimum vertical displacement
    u_prop_min : float
        Minimum propeller freestream inflow velocity
    energy : float
        Electrical energy consumed
    aoa_max : float
        Maximum effective angle of attack
    aoa_min : float
        Minimum effective angle of attack
    acc_max : float
        Maximum acceleration magnitude
    """

    def initialize(self):
        # declare the input dict provided in the run script
        self.options.declare('input_dict', types=dict)

    def setup(self):
        input_dict = self.options['input_dict']

        # give variable names to user-specified values from input dict
        self.x_dot_initial = input_dict['x_dot_initial'] # initial horizontal speed
        self.y_dot_initial = input_dict['y_dot_initial'] # initial vertical speed
        self.y_initial = input_dict['y_initial'] # initial vertical displacement
        self.A_disk = input_dict['A_disk'] # total propeller disk area
        self.T_guess = input_dict['T_guess'] # initial thrust guess
        self.alpha_stall = input_dict['alpha_stall'] # wing stall angle
        self.CD0 = input_dict['CD0'] # coefficient of drag of the fuselage, gear, etc.
        self.AR = input_dict['AR'] # aspect ratio
        self.e = input_dict['e'] # span efficiency factor of each wing
        self.rho = input_dict['rho'] # air density
        self.S = input_dict['S'] # total wing reference area
        self.m = input_dict['m'] # mass of aircraft
        self.a0 = input_dict['a0'] # airfoil lift-curve slope
        self.t_over_c = input_dict['t_over_c'] # airfoil thickness-to-chord ratio
        self.v_factor = input_dict['induced_velocity_factor'] # induced-velocity factor
        self.stall_option = input_dict['stall_option'] # stall option: 's' allows stall, 'ns' does not
        self.num_steps = input_dict['num_steps'] # number of time steps
        self.R = input_dict['R'] # propeller radius
        self.solidity = input_dict['solidity'] # propeller solidity
        self.omega = input_dict['omega'] # propeller angular speed
        self.prop_CD0 = input_dict['prop_CD0'] # CD0 for propeller profile power
        self.k_elec = input_dict['k_elec'] # factor for mechanical and electrical losses
        self.k_ind = input_dict['k_ind'] # factor for induced losses
        self.nB = input_dict['nB'] # number of blades per propeller
        self.bc = input_dict['bc'] # representative blade chord
        self.n_props = input_dict['n_props'] # number of propellers

        self.quartic_poly_coeffs, pts = give_curve_fit_coeffs(self.a0, self.AR, self.e)
        
+       # state variables on which the ODE depends
+       self.add_input('vx', units='m/s')
+       self.add_input('vy', units='m/s')

        # openmdao inputs to the component
-       self.add_input('powers', val = np.ones(self.num_steps, dtype=complex))
-       self.add_input('thetas', val = np.ones(self.num_steps, dtype=complex))
        self.add_input('power', shape=(1,))
        self.add_input('theta', shape=(1,))
-       self.add_input('flight_time', val = 15.*60)
        # openmdao outputs from the component
        self.add_output('x_dot', val = 1.)
        self.add_output('y_dot', val = 1.)
-       self.add_output('x', val = 1.)
-       self.add_output('y', val = 1.)
+       self.add_output('vx_dot', val = 1.)
+       self.add_output('vy_dot', val = 1.)
-       self.add_output('y_min', val = 0.)
-       self.add_output('u_prop_min', val = 0.)
-       self.add_output('energy', val = 1.)
        self.add_output('energy_dot', val = 1.)
        if self.stall_option == 'ns':
+           self.add_output('aoa', shape=(1,))
-           self.add_output('aoa_max', val = 10.)
-           self.add_output('aoa_min', val = -10.)
-       self.add_output('acc_max', val = 1.)
+       self.add_output('acc', val = 1.)

-       # some state variables
-       self.x_dots = np.ones(self.num_steps + 1, dtype=complex) # horizontal speeds
-       self.y_dots = np.ones(self.num_steps + 1, dtype=complex) # vertical speeds
-       self.thrusts = np.ones(self.num_steps + 1, dtype=complex) # thrusts
-       self.atov = np.ones(self.num_steps + 1, dtype=complex) # freestream angles to vertical
-       self.CL = np.zeros(self.num_steps + 1, dtype=complex) # wing lift coefficients
-       self.CD = np.zeros(self.num_steps + 1, dtype=complex) # wing drag coefficients
-       self.x = np.zeros(self.num_steps + 1, dtype=complex) # horizontal positions
-       self.y = np.zeros(self.num_steps + 1, dtype=complex) # vertical positions
-       self.energy = np.zeros(self.num_steps + 1, dtype=complex) # electrical energy consumed
-       self.aoa = np.zeros(self.num_steps, dtype=complex) # effective wing angles of attack
-       self.u_inf_prop = np.zeros(self.num_steps, dtype=complex) # freestream speeds normal to propeller disks 
-       self.v_i = np.zeros(self.num_steps, dtype=complex) # effective propeller-induced speeds seen by wings
-       self.acc = np.zeros(self.num_steps, dtype=complex) # acceleration magnitude in g's
-       self.a_x = np.zeros(self.num_steps, dtype=complex) # horizontal acceleration
-       self.a_y = np.zeros(self.num_steps, dtype=complex) # vertical acceleration
-       self.L_wings = np.zeros(self.num_steps, dtype=complex) # total lift of the wings
-       self.D_wings = np.zeros(self.num_steps, dtype=complex) # total drag of the wings
-       self.D_fuse = np.zeros(self.num_steps, dtype=complex) # drag of the fuselage
-       self.N = np.zeros(self.num_steps, dtype=complex) # total propeller normal force
-       self.aoa_prop = np.zeros(self.num_steps, dtype=complex) # propeller angle of attack

        # use complex step for partial derivatives
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):

-       self.x_dots[0] = self.x_dot_initial
-       self.y_dots[0] = self.y_dot_initial
-       thrust = self.T_guess
-       self.atov[0] = c_atan2(self.x_dots[0], self.y_dots[0])
-       self.energy[0] = 0.
-       self.y[0] = self.y_initial

        self.flight_time = inputs['flight_time']
-       self.powers = inputs['powers']
-       self.thetas = inputs['thetas']       
+       power = inputs['power']
+       theta = inputs['theta']
+       vx = inputs['vx']
+       vy = inputs['vy']

-       dt = self.flight_time / self.num_steps

-       # time integration
-       for i in range(self.num_steps):
-
-           x_dot_old = self.x_dots[i]
-           y_dot_old = self.y_dots[i]
-           power = self.powers[i]
-           theta = self.thetas[i]
-
-           # the freestream angle relative to the vertical is
-           atov = c_atan2(x_dot_old, y_dot_old)
-           # the freestream speed is
-           v_inf = (x_dot_old**2 + y_dot_old**2)**0.5
-
-           self.u_inf_prop[i] = v_inf * np.cos(atov - theta)
-           u_parallel = v_inf * np.sin(atov - theta)
-
-           mu = u_parallel / (self.omega * self.R)
-           CP_profile = self.solidity * self.prop_CD0 / 8. * (1 + 4.6 * mu**2)
-           P_disk = self.k_elec * power - CP_profile * (self.rho * self.A_disk * (self.omega * self.R)**3)
-
-           self.thrusts[i], vi = Thrust(self.u_inf_prop[i], P_disk, self.A_disk, thrust, self.rho, self.k_ind)
-           thrust = self.thrusts[i]
-           
-           Normal_F = Normal_force(v_inf, self.R, thrust / self.n_props, atov - theta, self.rho, self.nB, self.bc)
-
-           step = change(atov, v_inf, dt, theta, thrust, self.alpha_stall, self.CD0, self.AR, self.e, self.rho, self.S, self.m, self.a0, self.t_over_c, self.quartic_poly_coeffs, vi, self.v_factor, self.n_props * Normal_F)
-
-           self.x_dots[i+1] = self.x_dots[i] + step[0][0]
-           self.y_dots[i+1] = self.y_dots[i] + step[1][0]
-           self.acc[i] = ((step[0][0] / dt)**2 + (step[1][0] / dt)**2)**0.5 / 9.81
-           self.a_x[i] = step[0][0] / dt
-           self.a_y[i] = step[1][0] / dt
-           self.atov[i+1] = atov
-           self.CL[i+1] = step[2]
-           self.CD[i+1] = step[3]
-           self.x[i+1] = self.x[i] + self.x_dots[i] * dt[0]
-           self.y[i+1] = self.y[i] + self.y_dots[i] * dt[0]
-           self.energy[i+1] = self.energy[i] + power * dt[0]
-           self.v_i[i] = vi * self.v_factor
-           self.aoa[i] = step[4]
-           self.L_wings[i] = step[5]
-           self.D_wings[i] = step[6]
-           self.D_fuse[i] = step[7]
-           self.N[i] = self.n_props * Normal_F
-           self.aoa_prop[i] = atov - theta

        # the freestream angle relative to the vertical is
        atov = cs_atan2(vx, vy)

        # the freestream speed is
        v_inf = (vx ** 2 + vy ** 2) ** 0.5

        u_inf_prop = v_inf * np.cos(atov - theta)
        u_parallel = v_inf * np.sin(atov - theta)

        mu = u_parallel / (self.omega * self.R)
        CP_profile = self.solidity * self.prop_CD0 / 8. * (1 + 4.6 * mu ** 2)
        P_disk = self.k_elec * power - CP_profile * (self.rho * self.A_disk * (self.omega * self.R) ** 3)

        thrust, vi = Thrust(u_inf_prop, P_disk, self.A_disk, thrust, self.rho, self.k_ind)

        Normal_F = Normal_force(v_inf, self.R, thrust / self.n_props, atov - theta, self.rho, self.nB, self.bc)

        CL, CD, aoa_blown, L, D_wings, D_fuse = aero(atov, v_inf, theta, thrust, self.alpha_stall,
                                                     self.CD0, self.AR, self.e, self.rho, self.S,
                                                     self.m, self.a0, self.t_over_c,
                                                     self.quartic_poly_coeffs, vi, self.v_factor,
                                                     self.n_props * Normal_F)

-       outputs['x_dot'] = self.x_dots[i+1]
-       outputs['y_dot'] = self.y_dots[i+1]
-       outputs['x'] = self.x[i+1]
-       outputs['y'] = self.y[i+1]
-       outputs['energy'] = self.energy[i+1]
        
+       outputs['x_dot'] = vx
+       outputs['y_dot'] = vy
+       outputs['vx_dot'] = (thrust * np.sin(theta) - D_fuse * np.sin(atov) - D_wings * np.sin(theta + aoa_blown) - L * np.cos(theta + aoa_blown) - Normal_F * np.cos(theta)) / self.m
+       outputs['vy_dot'] = (thrust * np.cos(theta) - D_fuse * np.cos(atov) - D_wings * np.cos(theta + aoa_blown) + L * np.sin(theta + aoa_blown) + Normal_F * np.sin(theta) - self.m * 9.81) / self.m
+       outputs['energy_dot'] = power
-
-       # use KS function for minimum vertical displacement
-       ks_rho = 100. # Hard coded, see Martins and Poon 2005 for more (if using a larger number of time steps than ~500, this ks_rho value should be higher)
-       f_max = np.max(-self.y)
-       min_y = -(f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-self.y - f_max)))))
-       outputs['y_min'] = min_y
-
-       # use KS function for minimum propeller inflow speed
-       ks_rho = 100. # Hard coded, see Martins and Poon 2005 for more (if using a larger number of time steps than ~500, this ks_rho value should be higher)
-       f_max = np.max(-self.u_inf_prop)
-       self.min_u_prop = -(f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-self.u_inf_prop - f_max)))))
-       outputs['u_prop_min'] = self.min_u_prop
-
-       # use KS function for minimum and maximum effective angles of attack
-       if self.stall_option == 'ns':
-           ks_rho = 500. # Hard coded, see Martins and Poon 2005 for more
-           f_max = np.max(self.aoa)
-           max_aoa = (f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (self.aoa - f_max)))))
-           outputs['aoa_max'] = max_aoa
-
-           ks_rho = 500. # Hard coded, see Martins and Poon 2005 for more
-           f_max = np.max(-self.aoa)
-           min_aoa = -(f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-self.aoa - f_max)))))
-           outputs['aoa_min'] = min_aoa
-       
-       ks_rho = 500. # Hard coded, see Martins and Poon 2005 for more
-       f_max = np.max(self.acc)
-       max_acc = (f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (self.acc - f_max)))))
-       outputs['acc_max'] = max_acc
```

## Vectorization of the ODE

At each time-step, the ODE (as given) operates on _scalar_ inputs.
Dymos uses an implicit simulation technique.
It assumes the trajectory is known (the user starts with some initial guess of the trajectory) and each state-time history can be treated as a polynomial.
It then evaluates the ODE at various times across the trajectory (so-called _nodes_), and compares the difference between the slope of the state-time history polynomials and the outputs of the ODE.
It then changes the trajectory until the difference between the slopes and the computed time-derivatives (the so-called _defects_) match.
In order to do this, we need the ODE to be able to evaluate many points simultaneously.

To do this, ODE's used in Dymos must have an option (`num_nodes`) which tells them how many points are to be computed simultaneously.
This option is declared in the `initialize` method of the ODE Group/Components.

All inputs and outputs are then sized based on the number of nodes being evaluated.
Having implemented these changes, the Dynamics component is now:

```python
class Dynamics(om.ExplicitComponent):
    """
    This is the OpenMDAO component that takes the design variables and computes
    the objective function and other quantities of interest.

    Parameters
    ----------
    powers : array
        Electrical power distribution as a function of time
    thetas : array
        Wing-angle-to-vertical distribution as a function of time
    flight_time : float
        Duration of flight

    Returns
    -------
    x_dot : float
        Final horizontal speed
    y_dot : float
        Final vertical speed
    x : float
        Final horizontal position
    y : float
        Final vertical position
    y_min : float
        Minimum vertical displacement
    u_prop_min : float
        Minimum propeller freestream inflow velocity
    energy : float
        Electrical energy consumed
    aoa_max : float
        Maximum effective angle of attack
    aoa_min : float
        Minimum effective angle of attack
    acc_max : float
        Maximum acceleration magnitude
    """

    def initialize(self):
        # declare the input dict provided in the run script
        self.options.declare('input_dict', types=dict)
        self.options.declare('num_nodes', types=(int,), default=1)

    def setup(self):
        input_dict = self.options['input_dict']

        # # give variable names to user-specified values from input dict
        self.A_disk = input_dict['A_disk']  # total propeller disk area
        self.T_guess = input_dict['T_guess']  # initial thrust guess
        self.alpha_stall = input_dict['alpha_stall']  # wing stall angle
        self.CD0 = input_dict['CD0']  # coefficient of drag of the fuselage, gear, etc.
        self.AR = input_dict['AR']  # aspect ratio
        self.e = input_dict['e']  # span efficiency factor of each wing
        self.rho = input_dict['rho']  # air density
        self.S = input_dict['S']  # total wing reference area
        self.m = input_dict['m']  # mass of aircraft
        self.a0 = input_dict['a0']  # airfoil lift-curve slope
        self.t_over_c = input_dict['t_over_c']  # airfoil thickness-to-chord ratio
        self.v_factor = input_dict['induced_velocity_factor']  # induced-velocity factor
        self.stall_option = input_dict['stall_option']  # stall option: 's' allows stall, 'ns' does not
        # self.num_steps = input_dict['num_steps']  # number of time steps
        self.R = input_dict['R']  # propeller radius
        self.solidity = input_dict['solidity']  # propeller solidity
        self.omega = input_dict['omega']  # propeller angular speed
        self.prop_CD0 = input_dict['prop_CD0']  # CD0 for propeller profile power
        self.k_elec = input_dict['k_elec']  # factor for mechanical and electrical losses
        self.k_ind = input_dict['k_ind']  # factor for induced losses
        self.nB = input_dict['nB']  # number of blades per propeller
        self.bc = input_dict['bc']  # representative blade chord
        self.n_props = input_dict['n_props']  # number of propellers

        self.quartic_poly_coeffs, pts = give_curve_fit_coeffs(self.a0, self.AR, self.e)

        nn = self.options['num_nodes']
        self.add_input('power', val=np.ones(nn))
        self.add_input('theta', val=np.ones(nn))
        self.add_input('vx', val=np.ones(nn))
        self.add_input('vy', val=np.ones(nn))
        self.add_output('x_dot', val=np.ones(nn))
        self.add_output('y_dot', val=np.ones(nn))
        self.add_output('vx_dot', val=np.ones(nn))
        self.add_output('vy_dot', val=np.ones(nn))
        self.add_output('energy_dot', val=np.ones(nn))

        # use complex step for partial derivatives
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):

        thrust = self.T_guess * np.ones(self.options['num_nodes'])

        # self.flight_time = inputs['flight_time']
        power = inputs['power']
        theta = inputs['theta']

        vx = inputs['vx']
        vy = inputs['vy']

        # the freestream angle relative to the vertical is
        atov = cs_atan2(vx, vy)

        # the freestream speed is
        v_inf = (vx ** 2 + vy ** 2) ** 0.5

        u_inf_prop = v_inf * np.cos(atov - theta)
        u_parallel = v_inf * np.sin(atov - theta)

        mu = u_parallel / (self.omega * self.R)
        CP_profile = self.solidity * self.prop_CD0 / 8. * (1 + 4.6 * mu ** 2)
        P_disk = self.k_elec * power - CP_profile * (self.rho * self.A_disk * (self.omega * self.R) ** 3)

        thrust, vi = Thrust(u_inf_prop, P_disk, self.A_disk, thrust, self.rho, self.k_ind)

        Normal_F = Normal_force(v_inf, self.R, thrust / self.n_props, atov - theta, self.rho, self.nB, self.bc)

        CL, CD, aoa_blown, L, D_wings, D_fuse = aero(atov, v_inf, theta, thrust, self.alpha_stall,
                                                     self.CD0, self.AR, self.e, self.rho, self.S,
                                                     self.m, self.a0, self.t_over_c,
                                                     self.quartic_poly_coeffs, vi, self.v_factor,
                                                     self.n_props * Normal_F)

        # compute horizontal and vertical changes in velocity
        outputs['x_dot'] = vx
        outputs['y_dot'] = vy
        outputs['vx_dot'] = (thrust * np.sin(theta) - D_fuse * np.sin(atov) - D_wings * np.sin(theta + aoa_blown) - L * np.cos(theta + aoa_blown) - Normal_F * np.cos(theta)) / self.m
        outputs['vy_dot'] = (thrust * np.cos(theta) - D_fuse * np.cos(atov) - D_wings * np.cos(theta + aoa_blown) + L * np.sin(theta + aoa_blown) + Normal_F * np.sin(theta) - self.m * 9.81) / self.m
        outputs['energy_dot'] = power
```

## Vectorization of the supporting functions

The following changes also need to be made to vectorize the supporting functions in the original implementation.

### atan2
The complex-step-compatible arctangent method is rewritten as:

```python
def cs_atan2(y, x):
    """
    A numpy-compatible, complex-compatible arctan2 function for use with complex-step.

    Parameters
    ----------
    y : float or complex
        The length of the side opposite the angle being determined.
    x : float or complex
        The length of the side adjacent to the angle being determined.

    Returns
    -------
    The angle whose opposite side has length y and whose adjacent side has length x.
    """
    a = np.real(y)
    b = np.imag(y)
    c = np.real(x)
    d = np.imag(x)

    if np.any(np.iscomplex(x)) or np.any(np.iscomplex(y)):
        res = np.arctan2(a, c) + 1j * (c * b - a * d) / (a**2 + c**2)
    else:
        res = np.arctan2(a, c)
    return res
```

### Thrust

The thrust-computing function contains its own Newton solver.
However, the exit criteria is only valid for a scalar thrust.

```python
while np.abs(T_old.real - thrust.real) > 1e-10:
```

We can convert this to a vectorized form as:

```python
while np.any(np.abs(T_old - thrust) > 1e-10):
```

## CLfunc and CDfunc

These methods are a bit more complex.
Both contain an if-condition with slightly different behavior depending on the incoming value of alpha.
As a general warning, this can cause problems with differentiation near the breakpoint, as the user noted in their code:

```python
        # this is because something strange happens very near 0 for complex step
        if angle.real > -1e-8:
            CL = CLa * angle
```

Now that angle is vectorized, we need to find which indices are positive and which are negative.
As we perform the following calculations, we'll assign the results into the corresponding indices of the output.

```python
    pos_idxs = np.where(angle >= 0)[0]
    neg_idxs = np.where(angle < 0)[0]
    ...
    if np.any(angle >= 0):
        CL[pos_idxs] = -(fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CL_array - fmax[:, np.newaxis])))))
    if np.any(angle < 0):
        CL[neg_idxs] = (fmax_neg + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CL_array_neg - fmax_neg[:, np.newaxis])))))
```


```python
def CLfunc(angle, alpha_stall, AR, e, a0, t_over_c):
    """
    This gives the lift coefficient of a wing as a function of angle of attack.

    Parameters
    ----------
    angle : float
        Angle of attack
    alpha_stall : float
        Stall angle of attack
    AR : float
        Aspect ratio
    e : float
        Span efficiency factor
    a0 : float
        Airfoil lift-curve slope
    t_over_c : float
        Thickness-to-chord ratio

    Returns
    -------
    CL : float
        Lift coefficient of the wing
    """
    pos_idxs = np.where(angle >= 0)[0]
    neg_idxs = np.where(angle < 0)[0]
    CL = np.zeros_like(angle)

    CLa = a0 / (1 + a0 / (np.pi * e * AR))
    CL_stall = CLa * alpha_stall

    # CD_max = (1. + 0.065 * AR) / (0.9 + t_over_c)
    CD_max = 1.1 + 0.018 * AR

    CL1 = CLa * angle

    A1 = CD_max / 2
    A2 = (CL_stall - CD_max * np.sin(alpha_stall) * np.cos(alpha_stall)) * np.sin(alpha_stall) / np.cos(alpha_stall)**2
    CL2 = A1 * np.sin(2 * angle) + A2 * np.cos(angle)**2 / np.sin(angle)

    CL_array = np.vstack((-CL1, -CL2)).T
    CL_array_neg = np.vstack((CL1, CL2)).T

    ks_rho = 50. # Hard coded, see Martins and Poon 2005 for more

    # Get the max at each node
    fmax = np.max(CL_array, axis=1)
    fmax_neg = np.max(CL_array_neg, axis=1)

    if np.any(angle >= 0):
        CL[pos_idxs] = -(fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CL_array - fmax[:, np.newaxis])))))
    if np.any(angle < 0):
        CL[neg_idxs] = (fmax_neg + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CL_array_neg - fmax_neg[:, np.newaxis])))))

    return CL
```

CDfunc has similar changes.
We should also point out the use of the complex-safe form of absolute value `cs_abs`, available from `openmdao.utils.cs_safe`.

```python
def CDfunc(angle, AR, e, alpha_stall, coeffs, a0, t_over_c):
    """
    This gives the drag coefficient of a wing as a function of angle of attack.

    Parameters
    ----------
    angle : float
        Angle of attack
    AR : float
        Aspect ratio
    e : float
        Span efficiency factor
    alpha_stall : float
        Stall angle of attack
    coeffs : array
        Curve-fit polynomial coefficients for the drag coefficient below 27.5 deg
    a0 : float
        Airfoil lift-curve slope
    t_over_c : float
        Thickness-to-chord ratio

    Returns
    -------
    CD : float
        Drag coefficient of the wing
    """

    quartic_poly_coeffs = coeffs

    cla =  a0 / (1 + a0 / (np.pi * e * AR))
    cl_stall = cla * alpha_stall
    CD_stall = np.dot(quartic_poly_coeffs, np.array([1, alpha_stall**2, alpha_stall**4]))

    CD_max = (1. + 0.065 * AR) / (0.9 + t_over_c)

    B1 = CD_max
    B2 = (CD_stall - CD_max * np.sin(alpha_stall)) / np.cos(alpha_stall)

    ks_rho = 50. # Hard coded, see Martins and Poon 2005 for more

    abs_angle = cs_abs(angle)

    # this is for the first part (quartic fit)
    CD_pt1 = np.dot(quartic_poly_coeffs, np.array([1, (27.5/180.*np.pi)**2, (27.5/180.*np.pi)**4])) * np.ones_like(abs_angle)
    CD_pt2 = B1 * np.sin(28./180.*np.pi) + B2 * np.cos(28./180.*np.pi) * np.ones_like(abs_angle)

    adjustment_line = (CD_pt2 - CD_pt1) / (0.5/180.*np.pi) * (abs_angle - 28./180.*np.pi) + CD_pt2

    CD1 = np.dot(quartic_poly_coeffs, np.array([1, abs_angle**2, abs_angle**4]))

    CD_array = np.vstack([CD1, adjustment_line]).T
    fmax = np.max(CD_array, axis=1)
    CD2 = (fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CD_array - fmax[:, np.newaxis])))))

    # this is for the second part (Tangler--Ostowari)
    CD3 = B1 * np.sin(abs_angle) + B2 * np.cos(abs_angle)

    CD_array = np.vstack([CD3, CD_pt1]).T
    fmax = np.max(CD_array, axis=1)
    CD4 = (fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CD_array - fmax[:, np.newaxis])))))

    # this puts them together
    CD_array = np.vstack([-CD2, -CD4]).T
    fmax = np.max(CD_array, axis=1)
    CD4 = -(fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CD_array - fmax[:, np.newaxis])))))
    
    return CD4
```

## Aero

The original problem uses a `change` method to increment the state vector using Euler propagation.
That `change` method is where the aerodynamic forces were accumulated.
Since our ODE is no longer actually propagating the states, but just returning their rates, we'll break those calculations out into the `aero` method.

```python
def aero(atov, v_inf, theta, T, alpha_stall, CD0, AR, e, rho, S, m, a0, t_over_c, coeffs, v_i, v_factor, Normal_F):
    aoa = atov - theta
    v_chorwise = v_inf * np.cos(aoa)
    v_normal = v_inf * np.sin(aoa)

    v_chorwise += v_i * v_factor
    v_blown = (v_chorwise ** 2 + v_normal ** 2) ** 0.5
    aoa_blown = cs_atan2(v_normal, v_chorwise)

    CL = CLfunc(aoa_blown, alpha_stall, AR, e, a0, t_over_c)
    CD = CDfunc(aoa_blown, AR, e, alpha_stall, coeffs, a0, t_over_c)

    # compute lift and drag forces
    L = 0.5 * rho * v_blown ** 2 * CL * S
    D_wings = 0.5 * rho * v_blown ** 2 * CD * S
    D_fuse = 0.5 * rho * v_inf ** 2 * CD0 * S

    return CL, CD, aoa_blown, L, D_wings, D_fuse, aoa_blown
```

## The Dynamics Component

Finally we create the new Dynamics component.
The key differences are as follows:

* In the original implementation, this component performed euler integration and provided the final values of the states as outputs.
* This version is vectorized, it computes all values of the state rates throughout the presumed trajectory simultaneously and return the entire history.

* In the original version, auxiliary outputs were saved in attributes of the model.
* In this version, the auxiliary outputs are provided as outputs.

* Variables to be path constrained used a K-S function and returned the K-S filtered extrema to be constrained.
* Since the component no longer sees the entire time history, we output auxiliary variables and use Dymos path constraints to bound them.

```python
class Dynamics(om.ExplicitComponent):
    """
    This is the OpenMDAO component that takes the design variables and computes
    the objective function and other quantities of interest.

    Parameters
    ----------
    powers : array
        Electrical power distribution as a function of time
    thetas : array
        Wing-angle-to-vertical distribution as a function of time
    flight_time : float
        Duration of flight

    Returns
    -------
    x_dot : float
        Final horizontal speed
    y_dot : float
        Final vertical speed
    x : float
        Final horizontal position
    y : float
        Final vertical position
    y_min : float
        Minimum vertical displacement
    u_prop_min : float
        Minimum propeller freestream inflow velocity
    energy : float
        Electrical energy consumed
    aoa_max : float
        Maximum effective angle of attack
    aoa_min : float
        Minimum effective angle of attack
    acc_max : float
        Maximum acceleration magnitude
    """

    def initialize(self):
        # declare the input dict provided in the run script
        self.options.declare('input_dict', types=dict)
        self.options.declare('num_nodes', types=(int,), default=1)

    def setup(self):
        input_dict = self.options['input_dict']

        # # give variable names to user-specified values from input dict
        self.A_disk = input_dict['A_disk']  # total propeller disk area
        self.T_guess = input_dict['T_guess']  # initial thrust guess
        self.alpha_stall = input_dict['alpha_stall']  # wing stall angle
        self.CD0 = input_dict['CD0']  # coefficient of drag of the fuselage, gear, etc.
        self.AR = input_dict['AR']  # aspect ratio
        self.e = input_dict['e']  # span efficiency factor of each wing
        self.rho = input_dict['rho']  # air density
        self.S = input_dict['S']  # total wing reference area
        self.m = input_dict['m']  # mass of aircraft
        self.a0 = input_dict['a0']  # airfoil lift-curve slope
        self.t_over_c = input_dict['t_over_c']  # airfoil thickness-to-chord ratio
        self.v_factor = input_dict['induced_velocity_factor']  # induced-velocity factor
        self.stall_option = input_dict[
            'stall_option']  # stall option: 's' allows stall, 'ns' does not
        self.R = input_dict['R']  # propeller radius
        self.solidity = input_dict['solidity']  # propeller solidity
        self.omega = input_dict['omega']  # propeller angular speed
        self.prop_CD0 = input_dict['prop_CD0']  # CD0 for propeller profile power
        self.k_elec = input_dict['k_elec']  # factor for mechanical and electrical losses
        self.k_ind = input_dict['k_ind']  # factor for induced losses
        self.nB = input_dict['nB']  # number of blades per propeller
        self.bc = input_dict['bc']  # representative blade chord
        self.n_props = input_dict['n_props']  # number of propellers

        self.quartic_poly_coeffs, pts = give_curve_fit_coeffs(self.a0, self.AR, self.e)

        # openmdao inputs to the component
        nn = self.options['num_nodes']
        self.add_input('power', val=np.ones(nn))
        self.add_input('theta', val=np.ones(nn))
        self.add_input('vx', val=np.ones(nn))
        self.add_input('vy', val=np.ones(nn))

        # component outputs for the state rates
        self.add_output('x_dot', val=np.ones(nn))
        self.add_output('y_dot', val=np.ones(nn))
        self.add_output('a_x', val=np.ones(nn))
        self.add_output('a_y', val=np.ones(nn))
        self.add_output('energy_dot', val=np.ones(nn))

        # component outputs for auxiliary outputs we may want
        # to constrain or view in timeseries
        self.add_output('acc', val=np.ones(nn))
        self.add_output('CL', val=np.ones(nn))
        self.add_output('CD', val=np.ones(nn))
        self.add_output('L_wings', val=np.ones(nn))
        self.add_output('D_wings', val=np.ones(nn))
        self.add_output('atov', val=np.ones(nn))
        self.add_output('D_fuse', val=np.ones(nn))        
        self.add_output('aoa', val=np.ones(nn))
        self.add_output('thrust', val=np.ones(nn))
        self.add_output('vi', val=np.ones(nn))

        # use complex step for partial derivatives
        self.declare_partials('*', '*', method='fd')

        # Partial derivative coloring
        # self.declare_coloring(wrt=['*'], method='cs', tol=1.0E-12, num_full_jacs=5,
        #                       show_summary=True, show_sparsity=True, min_improve_pct=10.)

    def compute(self, inputs, outputs):

        thrust = self.T_guess * np.ones(self.options['num_nodes'])
        power = inputs['power']
        theta = inputs['theta']

        vx = inputs['vx']
        vy = inputs['vy']

        # the freestream angle relative to the vertical is
        atov = cs_atan2(vx, vy)

        # the freestream speed is
        v_inf = (vx ** 2 + vy ** 2) ** 0.5

        u_inf_prop = v_inf * np.cos(atov - theta)
        u_parallel = v_inf * np.sin(atov - theta)

        mu = u_parallel / (self.omega * self.R)
        CP_profile = self.solidity * self.prop_CD0 / 8. * (1 + 4.6 * mu ** 2)
        P_disk = self.k_elec * power - CP_profile * (self.rho * self.A_disk * (self.omega * self.R) ** 3)

        thrust, vi = Thrust(u_inf_prop, P_disk, self.A_disk, thrust, self.rho, self.k_ind)

        Normal_F = Normal_force(v_inf, self.R, thrust / self.n_props, atov - theta, self.rho, self.nB, self.bc)

        CL, CD, aoa_blown, L, D_wings, D_fuse, aoa_blown = aero(atov, v_inf, theta, thrust, self.alpha_stall,
                                                                self.CD0, self.AR, self.e, self.rho, self.S,
                                                                self.m, self.a0, self.t_over_c,
                                                                self.quartic_poly_coeffs, vi, self.v_factor,
                                                                self.n_props * Normal_F)

        # compute horizontal and vertical changes in velocity
        outputs['atov'] = atov
        outputs['x_dot'] = vx
        outputs['y_dot'] = vy
        outputs['a_x'] = (thrust * np.sin(theta) - D_fuse * np.sin(atov) -
                          D_wings * np.sin(theta + aoa_blown) - L * np.cos(theta + aoa_blown) -
                          self.n_props * Normal_F * np.cos(theta)) / self.m
        outputs['a_y'] = (thrust * np.cos(theta) - D_fuse * np.cos(atov) -
                          D_wings * np.cos(theta + aoa_blown) + L * np.sin(theta + aoa_blown)
                          + self.n_props * Normal_F * np.sin(theta) - self.m * 9.81) / self.m
        outputs['energy_dot'] = power

        outputs['acc'] = np.sqrt(outputs['a_x']**2 + outputs['a_y']**2) / 9.81
        outputs['CL'] = CL
        outputs['CD'] = CD
        outputs['L_wings'] = L
        outputs['D_wings'] = D_wings
        outputs['D_fuse'] = D_fuse
        outputs['aoa'] = aoa_blown
        outputs['thrust'] = thrust
        outputs['vi'] = vi
```
