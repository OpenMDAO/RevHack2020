# How to solve these eVTOL takeoff trajectory optimization problems with Dymos?

## Background:
* I implemented an approach to solve takeoff trajectory optimization problems for an eVTOL configuration with OpenMDAO.
* The approach uses explicit time integration, the complex-step derivative approximation, KS functions for constraint aggregation, OpenMDAO's B-spline component, and other such methods.
* The code can be found here https://bitbucket.org/shamsheersc19/tilt_wing_evtol_takeoff
* The script with the models is around 600 lines and the run-script is around 150 lines (Python 2; OpenMDAO v2.6.0). It also has a readme and comments explaining how to use it (please let me know if anything needs clarification).
* The models have simple analytic expressions.
* The journal paper (with optimization problem formulations and results) can be found here
https://www.researchgate.net/publication/337259571_Tilt-Wing_eVTOL_Takeoff_Trajectory_Optimization

## Request:
1) How do I implement this using Dymos?
2) What are the advantages over the current implementation?

## Stretch goals (or easier alternatives to the above request):
1) What all needs to be done to use this with the latest recommended versions of Python and OpenMDAO?
2) I solved these problems with little difficulty with SNOPT, but I didn't have the same success with SciPy SLSQP, which is a problem for others who may want to use this code. What can I do to make these problems converge with SciPy SLSQP (or any other freely available optimizer that you recommend)?
3) What are some poor practices that you observe, and what would you recommend instead and why?
