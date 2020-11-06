# Simulating an eVTOL takeoff and transition with (and without Dymos)

This problem falls into the [unsteady analysis][unsteady-class] class of optimization problems. 
The first thing that needs to be done when addressing these class of problems is to 
develop an ordinary differential equation that governs the dynamics. 

In this problem, an ODE for the eVTOL aircraft was provided by Shamsheer Chuahan. 
You can checkout the details in his paper [here](https://www.researchgate.net/publication/337259571_Tilt-Wing_eVTOL_Takeoff_Trajectory_Optimization). 
Shamsheer also provided his [original implementation](https://bitbucket.org/shamsheersc19/tilt_wing_evtol_takeoff), which included an Euler time integration built inside an OpenMDAO component. 

This problem served as a really good thought problem for the dev team for several reasons: 

1) The code uses complex-step derivatives, which allowed for a fairly complex set of calculations to be assembled into a single component. 
2) The time-step style time integration that was originally implemented is very familiar to most engineers, and hence its an obvious way to get your feet wet
3) there were 4 different ways we thought we could improve on the original implementation

# Our Solutions

* [dymos based using implicit integration](./evtol_dymos/evtol_dymos.py)
* [dymos based using implicit integration - Vectorized](./evtol_dymos/evtol_dymos_vectorized.py)
* [dymos based using shooting - Vectorized](./evtol_dymos/evtol_dymos_vectorized_shooting.py)
* [time-stepping with an RK4 integrator](./evtol_explicit_time_integration/time_step_rk4.py)

We also gave a detailed analysis in an [ipython notebook that you can mess with in google-colab](./evtol_dymos/evtol_dymos.ipynb)! 

## Note on the RK4 solution: 

This solution is incomplete. We wanted to replicate the complex-step through the integrator approach used in the original code, but found that OpenMDAO's problem API's didn't let us. 
We're working on fixing this, but for now ... its just an example of how do to time-stepping analysis in OpenMDAO. 

# Notes on the baseline code

One of the best features of the code provided was that it was extremely compact. 
There were exactly two files that we needed to look at, one engineering file and one run script. 
In total, less than 800 lines of code that included the ODE, the time integration, and the derivatives (complex-stepped)

The time-step style time integration meant the original implementation used an approach that optimal-control folks call a "shooting method" and MDO folks call "MDF". 

The original implementation took about 25 minutes to run.
That's not that long, and certainly well within what most people would consider reasonable time frame for optimization. 


# Keep the ODE and the integration separate
If there was one thing we absolutely would change about the original implementation it would be to separate the ODE code from the time integration code. 
Although for this problem, Euler integration worked fine there are lots of reasons why you might choose to move away from it in general. 
For almost any other higher order scheme, you'll need to make multiple calls to the ODE for a single time-step (see our RK implementation). 
So that the very least you would end up with the ODE separated out into its own function call, instead of mixed into the loop for the time integration. 

Another reason to enforce this separation is that it lets you switch time integration methods a lot more easily. 
This would give you a path to graduate from your time-stepping approaches to a Dymos based approach if you needed to in the future. 

# Our ODE 
We played around with a lot of different ways to handle the ODE, but ultimately settled on two versions: vectorized and non vectorized. In both cases, we assume an input of `num_nodes`. 
This is a good general practice, because it gives you a lot of flexibility. 
If you are planning to time-step, you can always set `num_nodes=1`, but later on set it to a larger number if you're trying out implicit schemes. 


* [Vectorized](./ode/evtol_dynamics_comp_vectorized.py) : using fast numpy element wise operations to speed up the calculations 
* [Non Vectorized](./ode/evtol_dynamics_comp.py) : using a normal for loop to iterate over the arrays in the ODE

The vectorized one is a bit faster here (with one caveat that we'll discuss below), so if you care about absolute speed this is the better choice. 
However, it is a bit trickier to code when you have any kind of conditional logic, so you might find the for-loop version an easier place to start. 

If you're new to python and unsure how vectorized numpy stuff works, I suggest always starting with the simpler for-loop code to get things working. 
Then you can write some tests to verify the values and work on upgrading later. 

# Complex-Step partial derivatives of vectorized ODES
The original implementation from Shamsheer used complex-step to compute partial derivatives, and it worked very well. 
The only downside to it was that it was very slow, because the full Euler time integration needed to be run for each complex-step call. 
However, the up side is that he saved a bunch of time and effort in not having to hand differentiate the 800 lines of code in his component. 
On balance, we judge this to be a fair trade... but we were able to do a good bit better using two tricks: 

## Trick 1: Partial Derivative Coloring 
OpenMDAO has some fancy graph-coloring features for both partial and total derivatives. 
In this case, the [partial derivative coloring](http://openmdao.org/twodocs/versions/3.4.0/features/experimental/simul_coloring_fd_cs.html) is the key feature. 
By removing the time-loop from the ODE, we create a structure where each element of the array is independent of the others. 

Here is what that looks like in terms of a sub-set of the partial derivative Jacobian: 

```
.....................f.................. 1  x_dot
......................f................. 2  x_dot
.......................f................ 3  x_dot
........................f............... 4  x_dot
.........................f.............. 5  x_dot
..........................f............. 6  x_dot
...........................f............ 7  x_dot
............................f........... 8  x_dot
.............................f.......... 9  x_dot
..............................f......... 10  y_dot
...............................f........ 11  y_dot
................................f....... 12  y_dot
.................................f...... 13  y_dot
..................................f..... 14  y_dot
...................................f.... 15  y_dot
....................................f... 16  y_dot
.....................................f.. 17  y_dot
......................................f. 18  y_dot
.......................................f 19  y_dot
f.........f.........f.........f......... 20  a_x
.f.........f.........f.........f........ 21  a_x
..f.........f.........f.........f....... 22  a_x
...f.........f.........f.........f...... 23  a_x
....f.........f.........f.........f..... 24  a_x
.....f.........f.........f.........f.... 25  a_x
......f.........f.........f.........f... 26  a_x
.......f.........f.........f.........f.. 27  a_x
........f.........f.........f.........f. 28  a_x
.........f.........f.........f.........f 29  a_x
f.........f.........f.........f......... 30  a_y
.f.........f.........f.........f........ 31  a_y
..f.........f.........f.........f....... 32  a_y
...f.........f.........f.........f...... 33  a_y
....f.........f.........f.........f..... 34  a_y
.....f.........f.........f.........f.... 35  a_y
......f.........f.........f.........f... 36  a_y
.......f.........f.........f.........f.. 37  a_y
........f.........f.........f.........f. 38  a_y
.........f.........f.........f.........f 39  a_y
````

### One caveat: be careful when you vectorize things

There were two versions of the ODE, one that used a for loop and one that was vectorized with numpy. 
Some small detail of the vectorization messed up the sparsity. 
You can see the dense block that shows up in the acceleration terms, which is traceable down to the CD function. 

We know the ODE give the right answers, because we set up a [test to check that](./ode/test_dynamics_comp.py). But some detail of the vectorization clearly broke the sparsity pattern. 

```
....................f................... 0  x_dot
.....................f.................. 1  x_dot
......................f................. 2  x_dot
.......................f................ 3  x_dot
........................f............... 4  x_dot
.........................f.............. 5  x_dot
..........................f............. 6  x_dot
...........................f............ 7  x_dot
............................f........... 8  x_dot
.............................f.......... 9  x_dot
..............................f......... 10  y_dot
...............................f........ 11  y_dot
................................f....... 12  y_dot
.................................f...... 13  y_dot
..................................f..... 14  y_dot
...................................f.... 15  y_dot
....................................f... 16  y_dot
.....................................f.. 17  y_dot
......................................f. 18  y_dot
.......................................f 19  y_dot
f.........ffffffffffffffffffffffffffffff 20  a_x
.f........ffffffffffffffffffffffffffffff 21  a_x
..f.......ffffffffffffffffffffffffffffff 22  a_x
...f......ffffffffffffffffffffffffffffff 23  a_x
....f.....ffffffffffffffffffffffffffffff 24  a_x
.....f....ffffffffffffffffffffffffffffff 25  a_x
......f...ffffffffffffffffffffffffffffff 26  a_x
.......f..ffffffffffffffffffffffffffffff 27  a_x
........f.ffffffffffffffffffffffffffffff 28  a_x
.........fffffffffffffffffffffffffffffff 29  a_x
f.........ffffffffffffffffffffffffffffff 30  a_y
.f........ffffffffffffffffffffffffffffff 31  a_y
..f.......ffffffffffffffffffffffffffffff 32  a_y
...f......ffffffffffffffffffffffffffffff 33  a_y
....f.....ffffffffffffffffffffffffffffff 34  a_y
.....f....ffffffffffffffffffffffffffffff 35  a_y
......f...ffffffffffffffffffffffffffffff 36  a_y
.......f..ffffffffffffffffffffffffffffff 37  a_y
........f.ffffffffffffffffffffffffffffff 38  a_y
.........fffffffffffffffffffffffffffffff 39  a_y
f....................................... 40  energy_dot
.f...................................... 41  energy_dot
..f..................................... 42  energy_dot
...f.................................... 43  energy_dot
....f................................... 44  energy_dot
.....f.................................. 45  energy_dot
......f................................. 46  energy_dot
.......f................................ 47  energy_dot
........f............................... 48  energy_dot
.........f.............................. 49  energy_dot
f.........ffffffffffffffffffffffffffffff 50  acc
.f........ffffffffffffffffffffffffffffff 51  acc
..f.......ffffffffffffffffffffffffffffff 52  acc
...f......ffffffffffffffffffffffffffffff 53  acc
....f.....ffffffffffffffffffffffffffffff 54  acc
.....f....ffffffffffffffffffffffffffffff 55  acc
......f...ffffffffffffffffffffffffffffff 56  acc
.......f..ffffffffffffffffffffffffffffff 57  acc
........f.ffffffffffffffffffffffffffffff 58  acc
.........fffffffffffffffffffffffffffffff 59  acc
..........f.........f.........f......... 60  CL
...........f.........f.........f........ 61  CL
............f.........f.........f....... 62  CL
.............f.........f.........f...... 63  CL
..............f.........f.........f..... 64  CL
...............f.........f.........f.... 65  CL
................f.........f.........f... 66  CL
.................f.........f.........f.. 67  CL
..................f.........f.........f. 68  CL
...................f.........f.........f 69  CL
..........ffffffffffffffffffffffffffffff 70  CD
..........ffffffffffffffffffffffffffffff 71  CD
..........ffffffffffffffffffffffffffffff 72  CD
..........ffffffffffffffffffffffffffffff 73  CD
..........ffffffffffffffffffffffffffffff 74  CD
..........ffffffffffffffffffffffffffffff 75  CD
..........ffffffffffffffffffffffffffffff 76  CD
..........ffffffffffffffffffffffffffffff 77  CD
..........ffffffffffffffffffffffffffffff 78  CD
..........ffffffffffffffffffffffffffffff 79  CD
..........f.........f.........f......... 80  L_wings
...........f.........f.........f........ 81  L_wings
............f.........f.........f....... 82  L_wings
.............f.........f.........f...... 83  L_wings
..............f.........f.........f..... 84  L_wings
...............f.........f.........f.... 85  L_wings
................f.........f.........f... 86  L_wings
.................f.........f.........f.. 87  L_wings
..................f.........f.........f. 88  L_wings
...................f.........f.........f 89  L_wings
..........ffffffffffffffffffffffffffffff 90  D_wings
..........ffffffffffffffffffffffffffffff 91  D_wings
..........ffffffffffffffffffffffffffffff 92  D_wings
..........ffffffffffffffffffffffffffffff 93  D_wings
..........ffffffffffffffffffffffffffffff 94  D_wings
..........ffffffffffffffffffffffffffffff 95  D_wings
..........ffffffffffffffffffffffffffffff 96  D_wings
..........ffffffffffffffffffffffffffffff 97  D_wings
..........ffffffffffffffffffffffffffffff 98  D_wings
..........ffffffffffffffffffffffffffffff 99  D_wings
```

## Trick 2: Implicit methods are more efficient because they are sparse

Both Shamsheer's original implementation and ours had for loops in the ODE. 
But ours shows significant sparsity while his doesn't. Why? 
Because the implicit methods that dymos uses (and the implicit style of shooting methods it uses too) 
intentionally create this kind of sparsity. 

If you're are going to use an all-times-in-memory approach, you can expect a lot more performance by using an implicit technique that relies on residuals to converge the time-series. 

If you want to know more details about this, check out our papers on [derivative coloring](http://openmdao.org/pubs/openmdao_bicoloring.pdf) and [Dymos](http://openmdao.org/pubs/falck_dymos_2019_scitech.pdf)


# Shooting VS. Implicit Methods

## Shooting Methods a.k.a MDF
In a shooting method, the optimizer gets to see the control schedule as a design variable. 
Then, given that control schedule the entire time history is simulated out and the objective and constraints can be evaluated. 

We can relate this in the MDO context to the Multidisciplinary Design Feasible (MDF) architecture. 
Normally MDF implies that some sort of solver is in place converging governing equations, and that the optimizer is being shown a **continuous feasible space** where those governing equations are solved. 

What are the governing equations in the context of an Euler integration, and what are the associated state variables? 
The state variables are the collection of discrete values that represent the state-time-history of your system. 
If you were modeling a simple cannonball problem, you might have 2 states: x and y. 
If you took 10 time steps then you'd have 10 state variables for x and 10 for y. 
So what then are the residuals? 
Assuming you have some ODE function like this: 

x_dot, y_dot = f(time, control, x, y)

You can define the i-th residual like this: 

R_xi = x_dot(time_i, control_i, x_i, y_i) * delta_time - x_i+1
R_yi = x_dot(time_i, control_i, x_i, y_i) * delta_time - y_i+1

You don't have to actually solve the time series as a big implicit system (though you can if you want to), but the residual exist none the less which makes the connection to the MDF architecture. 
Regardless whether you find time-history using an time-stepping approach or the implicit residual form, since the optimizer always see a fully complete time history its an MDF approach. 


## Implicit Collocation a.k.a SAND
In an implicit collocation method, you treat both the controls and the state time-history as design variables for the optimizer. 
Then, because you've now added a bunch of new degrees of freedom to the optimizer you also provide it new equality constraints (sometimes called defect-constraints in the optimal control world) that must also be satisfied. 
If you were using an Euler based time integration scheme, the defect constraints would be exactly the same as the residuals 
given above. 

In the MDO world, the defect constraints could also be called the governing equations of the system. 
When you give the governing equations to the optimizer as equality constraints that is called the Simultaneous Analysis and Design (SAND) architecture. 

The major practical change by using this approach is that the optimizer has a much larger space to navigate through, 
since it can now violate physics. 
This can be really helpful if you happen to have a problem with a non-continuous feasible space. 


## Which is better Shooting or Implicit?

There is no simple answer here. Generally speaking, implicit methods are faster and commonly find better answers. 
However, for some problems implicit methods can be numerically finicky and really sensitive to scaling of design vars and constraints. 
Shooting methods are easier to set up, and when they work are much less sensitive to scaling. 
However, they also have their own numerical challenges if there are any singularities in your ODE (e.g. if you divide by sin(angle-of-attack)) or if you have a bumpy non-contiguous search space. 

Both have uses, and we tested out several solutions to this problem to see what happened. 





[unsteady-class]: ../../solution_approaches/unsteady_analysis.md