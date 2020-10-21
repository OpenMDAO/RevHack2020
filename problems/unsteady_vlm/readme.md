# Original Request From @shamsheersc19

## Background:
* [OpenAeroStruct (OAS)][0] is gaining popularity and is often advertised as a good example of a practical application that takes advantage of the strengths of OpenMDAO.
* A few years ago, Giovanni Pesare implemented an unsteady VLM solver with OAS with a little direction from @shamsheersc19 (see attached thesis).
* In order to take advantage of OpenMDAO's analytic derivatives, the only clear path was to stamp out copies 
of the time-varying states for each time instance inside an OpenMDAO model. 
* This seems like a less than ideal way to set it up, are there better ways? 

# Structure of an unsteady problem

As Giovanni Pesare points out in his thesis, an unstead problem is made up of three parts: 

1) A block of computations before the unsteady part
2) The unsteady part
3) A block of computations after the unsteady part

The core of the proposed problem here is to examine the unsteady part of the problem. 
That is the part that is the toughest to fit into the OpenMDAO paradigm. 
Before we dive into that though, its important to consider the before and after blocks too. 
If you want to optimize things, you have to connect these three chunks of calculations together, 
and if you want to use gradient based optimization then you also have compute gradients across all three blocks. 

# The Unsteady Block 

The very first thing you have to do when tackling an unsteady problem is pinpoint what kind of unsteady problem you want to solve: 

## Unsteady Simulation
The unifying characteristic of unsteady simulation is that they lack any time-varying control that you want to optimize. 
This means that they are well defined mathematical problems that can fully solved given the boundary conditions. 

While you absolutely can wrap an optimizer around these problems (often to vary one of the boundary conditions), 
an optimizer is not necessary to solve them. 

### Initial Value Problem: 
Start with a known initial condition, and integrate to some fixed end time. 
You've probably used something like scipy's [solve_ivp][1] handle this kind of problem. 

Example: Given an initial angle and velocity, compute the trajectory of a cannonball. 


### Boundary value Problem 
Start with a known initial and final condition, and vary some parameter till the time history satisfies both of them. 

Example: Assuming an initial velocity, find the initial angle for the cannonball to be fired such that it travels 30 meters before hitting the ground. 


## Optimal Control Problem 
Mathematically, these problems can be posed as either IVP, BVP. 
I am drawing a distinction though, because these involve time-varying controls and hence often benefit from a different kind of time-integration approach. 
This makes them underdefined, so unlike simulation problems they can not be solved without an optimizer of some kind. 

Said another way, the presence of the time-varying control adds additional degrees of freedom so that (even if given all the boundary conditions) there is no longer a single unique solution. 
You must specify an objective that that will provide a unique solution. 

Example: Given a cannonball with an on board gimbaled thruster, assuming an initial velocity, find the initial angle for the cannonball to be fired, and the schedule of thrust and thrust-angle such that it travels 30 meters before hitting the ground, touches the ground with 0 velocity, and uses the least amount of fuel. 



# The problem posed here is a optimization around an unsteady simulation: 





[0]: github.com/mdolab/openaerostruct
[1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
