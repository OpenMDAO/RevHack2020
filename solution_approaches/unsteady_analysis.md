# Building unsteady/transient analysis in OpenMDAO: 

The OpenMDAO Dev team has built the [Dymos library][2] for doing transient analysis and optimization in OpenMDAO. 
You should definitely check this out, because it has a lot of features like multi-phase trajectories and higher order integration schemes. 
Dymos uses an implicit pseudo-spectral time integration approach that requires you to keep all times in memory at once. 

However, there is another time integration approach that you may be more familiar with. 
time-stepping schemes like Euler, trapazoidal, and RK4 use a for-loop like structure to cascade the states through time on step at a time. Something like this: 


## Two fundamental options for unsteady analysis

### All-times in memory at once

This is the approach that [Dymos][2] takes, though there are some other examples including [OpenConcept][openconcept] and [Ozone][ozone]. 
The primary motivation for this approach is that it is very fast when used in an optimization context --- see [Rob Falck's paper on Dymos][dymos paper], [Justin Gray's paper on coloring][coloring paper], or [John Hwang's paper on ozone][ozone paper] for some background on why its so fast. 
A secondary motivation is that it can help find better answers in a more numerically stable fashion. 

Some versions of the all-times in memory at once approach that resemble shooting methods --- where every point investigated includes a physically valid time simulation.
Others use an implicit formulation that relies on the optimizer to enforce physical consistency constraints so that the time history is only fully valid at the final converged optimum.

Regardless of which version you choose the key characteristic is that (shockingly) you need one copy of your ODE instantiated in memory for every time-point that is evaluated. 
In fast implementations, this usually means a vectorized ODE where every input, state variable, and output is an array of size `num_steps`. 

### Time-stepping 

Here is a quick example of time-stepping, using a simple for loop around an OpenMDAO problem. 
```python 

    import openmdao.api as om 

    p = om.Problem()

    p.model.add_subsystem('calc', om.ExecComp('dx_dt = 5*t'), promotes=['*'])

    p.setup()

    # assume a time-step of 1 second

    x =[0.,]
    t =[0,]
    delta_t = 1

    for i in range(10): 
        p['t'] = t[i]
        p.run_model()
        
        x.append(delta_t*p['dx_dt'][0])
        t.append(t[i]+1)  

    print(x)
    print(t)
```

The key practical benefit of this approach is that it removes some numerical challenges, because the simple for loop will always give an answer without the need for an optimizer to satisfy constraints or a solver to converge residuals. 
It may sometimes be a non meaningful answer if the integration scheme is unstable, but hopefully your optimizer can plow through that.

Another important benefit of this approach is that it uses a lot less memory. 
If you have just a few state variables and your ODE is relatively cheap then the memory usage isn't a concern. 
However if you ware looking to model an unsteady CFD phenomenon such as unsteady aerodynamics for an acoustic model, 
then you almost certainly can't afford to keep every time instance in memory at the same time! 

## Waterfall: a slow option you probably shouldn't choose 

When you wrap your head around OpenMDAO and its modular structure, some find it tempting to set up a time stepping structure directly inside an OpenMDAO group. 
They stamp out `n` copies of their ODE, and then use some intermediate components to compute state updates. 
This is exactly how the code in the [unsteady VLM][unsteady vlm] problem is set up. 

The nice thing about this structure is that it lets you leverage OpenMDAO's derivatives system to easily compute adjoint total derivatives across the time integration. 
That is very nice, but it comes at a high cost. 
You end up with something that is halfway in between the all-times and the time-stepping approaches because you have to keep everything in memory all at once but stuck with a slow sequential computation. 


## Which one should I choose? 

There is a clear need for both approaches, so we'll start by suggesting a structure for your code that will allow you to easily move from one to the other. 
You should make a stand-alone ODE component/group that essentially takes the form of `y_dot = f(t,x,y)` where `t` is time, `x` is a set of control/design variables and `y` is the array of state variables. 

This stand-alone group can then be used with Dymos, in a hand coded all-times implementation like OpenConcept, or in a time-stepping approach. 

The eVTOL trajectory optimization problem posed for RevHack 2020 was given to us in a waterfall like form. 
It wasn't a bunch of separate instances chained together in a group, but rather a hand coded euler integration loop inside a component, which used complex-step to compute derivatives across the entire calculation. 
We have to admit that this solution has some elegance to it, because it was extremely compact code wise and fairly easy to understand. 
It was not very fast though. Optimizations took between 20 and 50 minutes (depending on which case you ran). 

Our solutions (we tried a few different minor variations) used Dymos to handle the time integration, but retained the use of complex-step for the partial derivatives of the ODE itself. 
These solutions found the same answer as the original waterfall style solution, but took between 3 minutes and 30 seconds to optimize. 

### What about a time-stepping solution

We are working on this, and we'll come back and provide a solution that shows it. 
Our approach to this is different than the waterfall methods.
We'll use the same ODE class as in our Dymos based solutions, but we'll stick it into a [sub-problem][subproblem], wrap the time integration around that and then put that whole thing into its own component. 

We ran out of time to get this implemented, because we ran into a couple of complications with OpenMDAO V3.4. 
We wanted to retain the use of complex-step similar to the original solution method, but OpenMDAO V3.4 doesn't provide a problem-API level way to achieve that. 
So we're taking that on as a follow on effort once we add the feature to OpenMDAO. 
There was actually already a [POEM][cs-poem] up to propose that feature, even before RevHack, but its not quite ready to be accepted yet. 

Once that is ready, we'll add an example to OpenMDAO's docs on how to do this. 
We'll also add the time-stepping type integration to Dymos as well. 
So most users will just be able to rely on Dymos, but if we don't have the exact scheme you want or you don't want a dependency on an external library then you can work from the example to create your own.



[2]: https://github.com/OpenMDAO/dymos
[7]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
[ozone]: https://github.com/hwangjt/ozone
[openconcept]: https://github.com/mdolab/openconcept
[dymos paper]: http://openmdao.org/pubs/falck_dymos_2019_scitech.pdf
[coloring paper]: http://openmdao.org/pubs/openmdao_bicoloring.pdf
[ozone paper]: http://openmdao.org/pubs/hwang_munster_ode_2018.pdf
[unsteady vlm]: ../../problems/unsteady_vlm
[cs-poem]: https://github.com/OpenMDAO/POEMs/pull/66
[subproblem]: ../sub_problem.md