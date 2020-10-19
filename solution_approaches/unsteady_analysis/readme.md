# Building unsteady/transient analysis in OpenMDAO: 

The OpenMDAO Dev team has built the [Dymos library][2] for doing transient analysis and optimization in OpenMDAO. 
You should definitely check this out, because it has a lot of features like multi-phase trajectories and higher order integration schemes. 
Dymos uses an implicit pseudo-spectral time integration approach that requires you to keep all times in memory at once. 

However, there is another time integration approach that you may be more familiar with. 
time-stepping schemes like Euler, trapazoidal, and RK4 use a for-loop like structure to cascade the states through time on step at a time. 
The result is that you need to keep just one (or in some higher order schemes, just a few) time in memory at once. 
If you've used scipy's [solve_ivp][7] you are using a time-stepping method. 

Both all-time and time-stepping approaches have advantages and disadvantages. 
There are literally text books written about this! 
We're not going to pick sides, we'll just show you how to do both using the same OpenMDAO model. 

## Hackathon solutions that use the all-time approach 
* Create a Dymos implementation of an eVTOL takeoff optimization that already exists as a explicit time integration implementation in OpenMDAO
* Integrate an OpenAeroStruct analysis into a Dymos trajectory analysis model 

## Hackathon solutions that use the time-step approach 
* Build an unsteady VLM simulation using an OpenAeroStruct model as a base 
* Build a time-stepped eVTOL takeoff optimization that already exists as an all-time integration implementation in OpenMDAO


[2]: https://github.com/OpenMDAO/dymos
[7]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
