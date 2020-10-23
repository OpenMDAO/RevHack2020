# Different Approaches to Time Integration in OpenMDAO

* OpenMDAO is not, out-of-the-box, set up to to tackle optimization problems involving dynamics.
* Problem has been accepted as part of the reverse hackathon so that we can demonstrate multiple ways of approaching dynamic optimization problems in OpenMDAO

## Implementation using Dymos

[Dymos](https://github.com/OpenMDAO/dymos) is an OpenMDAO-based library for analysis and optimization of dynamic systems.
A user provides a set of ordinary differential equations (ODE) in the form of an OpenMDAO system.
Dymos can then be used to perform simple time integration of the system, or to optimize it using shooting methods or pseudospectral optimal control techniques.

## A nested-problem approach utilizing state-transformation matrices

A second approach used to tackle this integraiton will be to build a wrapper around a standard numerical integration tool.
But in general, numerical integrators do not allow one to propagate derivatives of the final state across the time integration.
How does one determine the sensitivity of the final state to the initial state and controls when using something like [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)?
Finite differencing can be utilized, but this means re-propagating the trajectory each time a design variable is perturbed, making the calculation extremely expensive.
In the approach used here, we will use OpenMDAO to propagate a "state transition matrix" along-side the user's equations of motion.
The state-transition matrix is a matrix which, multiplied by the initial state, yields the final state.

<a href="https://www.codecogs.com/eqnedit.php?latex=\bar{x}_f&space;=&space;\left[&space;\phi&space;\right]&space;\bar{x}_0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bar{x}_f&space;=&space;\left[&space;\phi&space;\right]&space;\bar{x}_0" title="\bar{x}_f = \left[ \phi \right] \bar{x}_0" /></a>

That is, the state transition matrix ($\phi$) is the jacobian matrix of the final state w.r.t. the initial state.
To integrate the state transition matrix, we start with a state transition matrix equal to the identity matrix at the initial time, and use the following formula as its derivative:

<a href="https://www.codecogs.com/eqnedit.php?latex=\left[&space;\dot{\phi}&space;\right]&space;=&space;f_x&space;\left[&space;\phi&space;\right]" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\left[&space;\dot{\phi}&space;\right]&space;=&space;f_x&space;\left[&space;\phi&space;\right]" title="\left[ \dot{\phi} \right] = f_x \left[ \phi \right]" /></a>

An excellent discussion of the state transition matrix is provided by [Pellegrini and Russell](https://www.researchgate.net/publication/281440699_On_the_Accuracy_of_Trajectory_State-Transition_Matrices).

## Implementation using a manually-coded numerical integration technique

A third option is to implement a numerical propagation technique as a for loop within a component and propagate the state and the derivatives through each step.

## Impact of Variable Time-Step Integraiton

Dymos uses an implicit approach that assumes that the dynamics can be satisfied given a discretizaiton (the collocation grid).
The other two approaches tested here should provide the behavior of an explicit time-integration.
That is, if the time-step is insufficient, there will be some error in the propagation, but the derivatives across the propagation will hold true.
Unfortunately, as shown by Pellegrini and Russell, this is not the case for the state transition matrix approach.
Higher-order terms arise due to the changing of the timestep that are not captured by the state transition matrix.
One can either accept these errors, or resort to fixed-time-step integraiton techniques.

# The eVTOL Trajectory Optimization Problem

Originally proposed by [shamsheersc19](https://github.com/shamsheersc19)

## Background

* This is the trajectory optimization of an electric vertical take-off and landing (eVTOL) aircraft with OpenMDAO.
* The original implementation can be found [here](https://bitbucket.org/shamsheersc19/tilt_wing_evtol_takeoff).
* The problem formulation and results are published [here](https://www.researchgate.net/publication/337259571_Tilt-Wing_eVTOL_Takeoff_Trajectory_Optimization)
* The model, as given, uses complex-step derivative approximation.

## Request

1) How can this problem be implemented in Dymos?
2) What are the advantages over the current implementation?

## Stretch goals (or easier alternatives to the above request):
1) What all needs to be done to use this with the latest recommended versions of Python and OpenMDAO?
2) I solved these problems with little difficulty with SNOPT, but I didn't have the same success with SciPy SLSQP, which is a problem for others who may want to use this code. What can I do to make these problems converge with SciPy SLSQP (or any other freely available optimizer that you recommend)?
3) What are some poor practices that you observe, and what would you recommend instead and why?
