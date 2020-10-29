# Different Approaches to Time Integration in OpenMDAO

* OpenMDAO is not, out-of-the-box, set up to to tackle optimization problems involving dynamics.
* Problem has been accepted as part of the reverse hackathon so that we can demonstrate multiple ways of approaching dynamic optimization problems in OpenMDAO

## Implementation using Dymos

[Dymos](https://github.com/OpenMDAO/dymos) is an OpenMDAO-based library for analysis and optimization of dynamic systems.
A user provides a set of ordinary differential equations (ODE) in the form of an OpenMDAO system.
Dymos can then be used to perform simple time integration of the system, or to optimize it using shooting methods or pseudospectral optimal control techniques.

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
