# --------------------------------------------------------------------------------------------------
# This contains the functions and OpenMDAO component class for the simplified eVTOL flight physics.
# This works at least with OpenMDAO version 2.6.0.
# SI units used for everthing unless otherwise specified.
# Author: Shamsheer Chauhan
# --------------------------------------------------------------------------------------------------

from __future__ import division, print_function
import numpy as np
from openmdao.api import ExplicitComponent

def c_atan2(x,y):
    """ This is an arctan2 function that works with the complex-step method."""
    a=x.real
    b=x.imag
    c=y.real
    d=y.imag
    return complex(np.arctan2(a,c),(c*b-a*d)/(a**2+c**2))

def give_curve_fit_coeffs(a0, AR, e):
    """
    This gives the coefficients for the quartic least-squares curve fit that is used for each wing's
    coefficient of drag below 27.5 deg.

    Parameters
    ----------
    a0 : float
        Airfoil lift-curve slope in 1/rad
    AR : float
        Aspect ratio
    e : float
        Span efficiency factor

    Returns
    -------
    quartic_poly_coeffs : array
        Coefficients of the curve fit
    data_pts : array
        Data points that are fitted
    """

    cla =  a0 / (1 + a0/(np.pi * e * AR))
    
    data_pts = np.array([[16. / 180. * np.pi, 0.1], # Tangler--Ostowari points
                        [20. / 180. * np.pi, 0.175], # Tangler--Ostowari points
                        [25. / 180. * np.pi, 0.275], # Tangler--Ostowari points
                        [27.5 / 180. * np.pi, 0.363], # Tangler--Ostowari points
                        [12. / 180. * np.pi, 0.015 + (cla * 12. / 180. * np.pi)**2 / np.pi / AR / e],
                        [10. / 180. * np.pi, 0.012 + (cla * 10. / 180. * np.pi)**2 / np.pi / AR / e],
                        [8. / 180. * np.pi, 0.0095 + (cla * 8. / 180. * np.pi)**2 / np.pi / AR / e],
                        [6. / 180. * np.pi, 0.008 + (cla * 6. / 180. * np.pi)**2 / np.pi / AR / e],
                        [4. / 180. * np.pi, 0.007 + (cla * 4. / 180. * np.pi)**2 / np.pi / AR / e],
                        [2. / 180. * np.pi, 0.0062 + (cla * 2. / 180. * np.pi)**2 / np.pi / AR / e],
                        [0. / 180. * np.pi, 0.006]])

    new_fit_matrix = np.array([[1, data_pts[0,0]**2, data_pts[0,0]**4 ],
                               [1, data_pts[1,0]**2, data_pts[1,0]**4 ],
                               [1, data_pts[2,0]**2, data_pts[2,0]**4 ],
                               [1, data_pts[3,0]**2, data_pts[3,0]**4 ],
                               [1, data_pts[4,0]**2, data_pts[4,0]**4 ],
                               [1, data_pts[5,0]**2, data_pts[5,0]**4 ],
                               [1, data_pts[6,0]**2, data_pts[6,0]**4 ],
                               [1, data_pts[7,0]**2, data_pts[7,0]**4 ],
                               [1, data_pts[8,0]**2, data_pts[8,0]**4 ],
                               [1, data_pts[9,0]**2, data_pts[9,0]**4 ],
                               [1, data_pts[10,0]**2, data_pts[10,0]**4 ]])

    quartic_poly_coeffs = np.linalg.solve(np.dot(new_fit_matrix.T, new_fit_matrix), np.dot(new_fit_matrix.T, data_pts[:, 1]))

    return quartic_poly_coeffs, data_pts

def Thrust(u0, power, A, T, rho, kappa):
    """
    This computes the thrust and induced velocity at the propeller disk.
    This uses formulas from propeller momentum theory.

    Parameters
    ----------
    u0 : float
        Freestream speed normal to the propeller disk
    power : float
        Power supplied to the propeller disk
    A : float
        Propeller disk area
    T : float
        Thrust guess
    rho : float
        Air density
    kappa: float
        Corection factor for non-uniform inflow and tip effects

    Returns
    -------
    thrust : float
        Thrust
    v_i : float
        Induced velocity at the propeller disk
    """

    T_old = T + 10.
    thrust = T

    # iteration loop to solve for the thrust as a function of power
    while np.abs(T_old.real - thrust.real) > 1e-10:
        T_old = thrust

        ### FPI (Fixed point iteration)
        # T_new = power / (u0 + kappa * (-u0/2 + 0.5 * (u0**2 + 2 * thrust / rho / A)**0.5))
        # thrust = thrust + (T_new - thrust) * 0.5

        # Newton-Raphson
        root_term = (u0**2 + 2 * thrust / rho / A)**0.5
        R = power - thrust * (u0 + kappa * (-u0 / 2 + 0.5 * root_term))
        R_prime = -u0 - kappa * ( -u0 / 2 + 0.5 * root_term + 0.5 * thrust / rho / A / root_term)
        thrust = T_old - R / R_prime

    # the induced velocity (i.e., velocity added at the disk) is
    v_i = (-u0 / 2 + (u0**2 / 4. + thrust / 2 / rho / A)**0.5)

    if u0.real < 0:
        # This model is not valid if the freestream speed is negative
        print("FREESTREAM SPEED IS NEGATIVE!", thrust, v_i)

    return thrust, v_i

def Normal_force(u0, radius, thrust, alpha, rho, nB, bc):
    """
    This computes the normal force developed by each propeller due to the incidence angle of the flow.
    These equations are from "Propeller at high incidence" by de Young, 1965.

    Parameters
    ----------
    u0 : float
        Freestream speed
    radius : float
        Propeller radius
    thrust : float
        Propeller thrust
    alpha: float
        Incidence angle
    rho : float
        Air density
    nB : float
        Number of blades
    bc : float
        Effective blade chord

    Returns
    -------
    normal_force : float
        Normal force generated by one propeller
    """

    # conversion factor to convert from m to ft becasue these emperical formulas use imperial units
    m2f = 3.28

    # propeller 0.75R pitch angle as a function of freestream
    beta = 10 + u0 / 67. * 25

    u0 = u0 * m2f * np.cos(alpha)
    Diam = 2 * radius * m2f
    rho = rho * 0.00194
    c = bc * m2f

    q = 0.5 * rho * u0**2
    A_d = np.pi * Diam**2 / 4.
    Tc = thrust / q / A_d
    f = 1 + 0.5 * ((1 + Tc)**.5 - 1) + Tc / 4. / (2 + Tc)
    sigma = 4 * nB / 3 / np.pi * c / Diam
    slope = 4.25 * sigma / (1 + 2 * sigma) * np.sin(beta/180.*np.pi + 8./180.*np.pi) * f * q * A_d

    normal_force = slope * np.tan(alpha) / 2.2046 * 9.81

    return normal_force

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

    if angle.real >=0:

        CLa = a0 / (1 + a0 / (np.pi * e * AR))
        CL_stall = CLa * alpha_stall

        # CD_max = (1. + 0.065 * AR) / (0.9 + t_over_c)
        CD_max = 1.1 + 0.018 * AR

        CL1 = CLa * angle

        A1 = CD_max / 2
        A2 = (CL_stall - CD_max * np.sin(alpha_stall) * np.cos(alpha_stall)) * np.sin(alpha_stall) / np.cos(alpha_stall)**2
        CL2 = A1 * np.sin(2 * angle) + A2 * np.cos(angle)**2 / np.sin(angle)

        CL_array = np.array([-CL1, -CL2], dtype = complex)
        ks_rho = 50. # Hard coded, see Martins and Poon 2005 for more
        fmax = np.max(CL_array)
        CL = -(fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CL_array - fmax)))))

        # this is because something strange happens very near 0 for complex step
        if angle.real < 1e-8:
            CL = CLa * angle

    else:

        CLa = a0 / (1 + a0 / (np.pi * e * AR))
        CL_stall = CLa * alpha_stall

        # CD_max = (1. + 0.065 * AR) / (0.9 + t_over_c)
        CD_max = 1.1 + 0.018 * AR

        CL1 = CLa * angle

        A1 = CD_max / 2
        A2 = (CL_stall - CD_max * np.sin(alpha_stall) * np.cos(alpha_stall)) * np.sin(alpha_stall) / np.cos(alpha_stall)**2
        CL2 = A1 * np.sin(2 * angle) + A2 * np.cos(angle)**2 / np.sin(angle)

        CL_array = np.array([CL1, CL2], dtype = complex)
        ks_rho = 50. # Hard coded, see Martins and Poon 2005 for more
        fmax = np.max(CL_array)
        CL = (fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CL_array - fmax)))))

        # this is because something strange happens very near 0 for complex step
        if angle.real > -1e-8:
            CL = CLa * angle

    return CL

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

    if angle.real < 0:
        angle = -angle

    # this is for the first part (quartic fit)
    CD_pt1 = np.dot(quartic_poly_coeffs, np.array([1, (27.5/180.*np.pi)**2, (27.5/180.*np.pi)**4]))
    CD_pt2 = B1 * np.sin(28./180.*np.pi) + B2 * np.cos(28./180.*np.pi)

    adjustment_line = (CD_pt2 - CD_pt1) / (0.5/180.*np.pi) * (angle - 28./180.*np.pi) + CD_pt2

    CD1 = np.dot(quartic_poly_coeffs, np.array([1, angle**2, angle**4]))

    CD_array = np.array([CD1, adjustment_line], dtype = complex)
    ks_rho = 50. # Hard coded, see Martins and Poon 2005 for more
    fmax = np.max(CD_array)
    CD2 = (fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CD_array - fmax)))))

    # this is for the second part (Tangler--Ostowari)
    CD3 = B1 * np.sin(angle) + B2 * np.cos(angle)

    CD_array = np.array([CD3, CD_pt1], dtype = complex)
    ks_rho = 50. # Hard coded, see Martins and Poon 2005 for more
    fmax = np.max(CD_array)
    CD4 = (fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CD_array - fmax)))))

    # this puts them together
    CD_array = np.array([-CD2, -CD4], dtype = complex)
    ks_rho = 50. # Hard coded, see Martins and Poon 2005 for more
    fmax = np.max(CD_array)
    CD4 = -(fmax + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (CD_array - fmax)))))

    return CD4

def change(atov, v_inf, dt, theta, T, alpha_stall, CD0, AR, e, rho, S, m, a0, t_over_c, coeffs, v_i, v_factor, Normal_F):
    """
    This computes the change in velocity for each time step.

    Parameters
    ----------
    atov : float
        Freestream angle to the vertical
    v_inf : float
        Freestream speed
    dt : float
        Time step size
    theta : float
        Wing angle to the vertical
    T : float
        Thrust
    alpha_stall : float
        Stall angle of attack
    CD0 : float
        Parasite drag coefficient for fuse, gear, etc.
    AR : float
        Aspect ratio
    e : float
        Span efficiency factor
    rho : float
        Air density
    S : float
        Wing planform area
    m : float
        Mass of the aircraft
    a0 : float
        Airfoil lift-curve slope
    t_over_c : float
        Thickness-to-chord ratio
    coeffs : array
        Curve-fit polynomial coefficients for the drag coefficient below 27.5 deg
    v_i : float
        Induced-velocity value from the propellers
    v_factor : float
        Induced-velocity factor
    Normal_F : float
        Total propeller forces normal to the propeller axes

    Returns
    -------
    delta_xdot : float
        Change in horizontal velocity
    delta_ydot : float
        Change in vertical velocity
    CL : float
        Wing lift coefficient
    CD : float
        Wing drag coefficient
    aoa_blown : float
        Effective angle of attack with prop wash
    L : float
        Total lift force of the wings
    D_wings : float
        Total drag force of the wings
    D_fuse : float
        Drag force of the fuselage
    """

    # use angle of attack of wing to estimate CL and CD
    aoa = atov - theta
    v_chorwise =  v_inf * np.cos(aoa)
    v_normal = v_inf * np.sin(aoa)

    v_chorwise += v_i * v_factor
    v_blown = (v_chorwise**2 + v_normal**2)**0.5
    aoa_blown = c_atan2(v_normal, v_chorwise)

    CL = CLfunc(aoa_blown, alpha_stall, AR, e, a0, t_over_c)
    CD = CDfunc(aoa_blown, AR, e, alpha_stall, coeffs, a0, t_over_c)

    # compute lift and drag forces
    L = 0.5 * rho * v_blown**2 * CL * S
    D_wings = 0.5 * rho * v_blown**2 * CD * S
    D_fuse = 0.5 * rho * v_inf**2 * CD0 * S

    # compute horizontal and vertical changes in velocity
    delta_xdot = (T * np.sin(theta) - D_fuse * np.sin(atov) - D_wings * np.sin(theta + aoa_blown) - L * np.cos(theta + aoa_blown) - Normal_F * np.cos(theta)) / m * dt
    delta_ydot = (T * np.cos(theta) - D_fuse * np.cos(atov) - D_wings * np.cos(theta + aoa_blown) + L * np.sin(theta + aoa_blown) + Normal_F * np.sin(theta) - m * 9.81) / m * dt

    return np.array([delta_xdot, delta_ydot, CL, CD, aoa_blown, L, D_wings, D_fuse])

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

        # openmdao inputs to the component
        self.add_input('powers', val = np.ones(self.num_steps, dtype=complex))
        self.add_input('thetas', val = np.ones(self.num_steps, dtype=complex))
        self.add_input('flight_time', val = 15.*60)
        # openmdao outputs from the component
        self.add_output('x_dot', val = 1.)
        self.add_output('y_dot', val = 1.)
        self.add_output('x', val = 1.)
        self.add_output('y', val = 1.)
        self.add_output('y_min', val = 0.)
        self.add_output('u_prop_min', val = 0.)
        self.add_output('energy', val = 1.)
        if self.stall_option == 'ns':
            self.add_output('aoa_max', val = 10.)
            self.add_output('aoa_min', val = -10.)
        self.add_output('acc_max', val = 1.)

        # some state variables
        self.x_dots = np.ones(self.num_steps + 1, dtype=complex) # horizontal speeds
        self.y_dots = np.ones(self.num_steps + 1, dtype=complex) # vertical speeds
        self.thrusts = np.ones(self.num_steps + 1, dtype=complex) # thrusts
        self.atov = np.ones(self.num_steps + 1, dtype=complex) # freestream angles to vertical
        self.CL = np.zeros(self.num_steps + 1, dtype=complex) # wing lift coefficients
        self.CD = np.zeros(self.num_steps + 1, dtype=complex) # wing drag coefficients
        self.x = np.zeros(self.num_steps + 1, dtype=complex) # horizontal positions
        self.y = np.zeros(self.num_steps + 1, dtype=complex) # vertical positions
        self.energy = np.zeros(self.num_steps + 1, dtype=complex) # electrical energy consumed
        self.aoa = np.zeros(self.num_steps, dtype=complex) # effective wing angles of attack
        self.u_inf_prop = np.zeros(self.num_steps, dtype=complex) # freestream speeds normal to propeller disks 
        self.v_i = np.zeros(self.num_steps, dtype=complex) # effective propeller-induced speeds seen by wings
        self.acc = np.zeros(self.num_steps, dtype=complex) # acceleration magnitude in g's
        self.a_x = np.zeros(self.num_steps, dtype=complex) # horizontal acceleration
        self.a_y = np.zeros(self.num_steps, dtype=complex) # vertical acceleration
        self.L_wings = np.zeros(self.num_steps, dtype=complex) # total lift of the wings
        self.D_wings = np.zeros(self.num_steps, dtype=complex) # total drag of the wings
        self.D_fuse = np.zeros(self.num_steps, dtype=complex) # drag of the fuselage
        self.N = np.zeros(self.num_steps, dtype=complex) # total propeller normal force
        self.aoa_prop = np.zeros(self.num_steps, dtype=complex) # propeller angle of attack

        # use complex step for partial derivatives
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):

        self.x_dots[0] = self.x_dot_initial
        self.y_dots[0] = self.y_dot_initial
        thrust = self.T_guess
        self.atov[0] = c_atan2(self.x_dots[0], self.y_dots[0])
        self.energy[0] = 0.
        self.y[0] = self.y_initial

        self.flight_time = inputs['flight_time']
        self.powers = inputs['powers']
        self.thetas = inputs['thetas']

        dt = self.flight_time / self.num_steps

        # time integration
        for i in range(self.num_steps):

            x_dot_old = self.x_dots[i]
            y_dot_old = self.y_dots[i]
            power = self.powers[i]
            theta = self.thetas[i]

            # the freestream angle relative to the vertical is
            atov = c_atan2(x_dot_old, y_dot_old)
            # the freestream speed is
            v_inf = (x_dot_old**2 + y_dot_old**2)**0.5

            self.u_inf_prop[i] = v_inf * np.cos(atov - theta)
            u_parallel = v_inf * np.sin(atov - theta)

            mu = u_parallel / (self.omega * self.R)
            CP_profile = self.solidity * self.prop_CD0 / 8. * (1 + 4.6 * mu**2)
            P_disk = self.k_elec * power - CP_profile * (self.rho * self.A_disk * (self.omega * self.R)**3)

            self.thrusts[i], vi = Thrust(self.u_inf_prop[i], P_disk, self.A_disk, thrust, self.rho, self.k_ind)
            thrust = self.thrusts[i]
            
            Normal_F = Normal_force(v_inf, self.R, thrust / self.n_props, atov - theta, self.rho, self.nB, self.bc)

            step = change(atov, v_inf, dt, theta, thrust, self.alpha_stall, self.CD0, self.AR, self.e, self.rho, self.S, self.m, self.a0, self.t_over_c, self.quartic_poly_coeffs, vi, self.v_factor, self.n_props * Normal_F)

            self.x_dots[i+1] = self.x_dots[i] + step[0][0]
            self.y_dots[i+1] = self.y_dots[i] + step[1][0]
            self.acc[i] = ((step[0][0] / dt)**2 + (step[1][0] / dt)**2)**0.5 / 9.81
            self.a_x[i] = step[0][0] / dt
            self.a_y[i] = step[1][0] / dt
            self.atov[i+1] = atov
            self.CL[i+1] = step[2]
            self.CD[i+1] = step[3]
            self.x[i+1] = self.x[i] + self.x_dots[i] * dt[0]
            self.y[i+1] = self.y[i] + self.y_dots[i] * dt[0]
            self.energy[i+1] = self.energy[i] + power * dt[0]
            self.v_i[i] = vi * self.v_factor
            self.aoa[i] = step[4]
            self.L_wings[i] = step[5]
            self.D_wings[i] = step[6]
            self.D_fuse[i] = step[7]
            self.N[i] = self.n_props * Normal_F
            self.aoa_prop[i] = atov - theta

        outputs['x_dot'] = self.x_dots[i+1]
        outputs['y_dot'] = self.y_dots[i+1]
        outputs['x'] = self.x[i+1]
        outputs['y'] = self.y[i+1]
        outputs['energy'] = self.energy[i+1]

        # use KS function for minimum vertical displacement
        ks_rho = 100. # Hard coded, see Martins and Poon 2005 for more (if using a larger number of time steps than ~500, this ks_rho value should be higher)
        f_max = np.max(-self.y)
        min_y = -(f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-self.y - f_max)))))
        outputs['y_min'] = min_y

        # use KS function for minimum propeller inflow speed
        ks_rho = 100. # Hard coded, see Martins and Poon 2005 for more (if using a larger number of time steps than ~500, this ks_rho value should be higher)
        f_max = np.max(-self.u_inf_prop)
        self.min_u_prop = -(f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-self.u_inf_prop - f_max)))))
        outputs['u_prop_min'] = self.min_u_prop

        # use KS function for minimum and maximum effective angles of attack
        if self.stall_option == 'ns':
            ks_rho = 500. # Hard coded, see Martins and Poon 2005 for more
            f_max = np.max(self.aoa)
            max_aoa = (f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (self.aoa - f_max)))))
            outputs['aoa_max'] = max_aoa

            ks_rho = 500. # Hard coded, see Martins and Poon 2005 for more
            f_max = np.max(-self.aoa)
            min_aoa = -(f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (-self.aoa - f_max)))))
            outputs['aoa_min'] = min_aoa
        
        ks_rho = 500. # Hard coded, see Martins and Poon 2005 for more
        f_max = np.max(self.acc)
        max_acc = (f_max + 1 / ks_rho * np.log(np.sum(np.exp(ks_rho * (self.acc - f_max)))))
        outputs['acc_max'] = max_acc
