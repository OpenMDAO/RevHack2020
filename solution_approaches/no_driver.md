# Using OpenMDAO's Driver interface is not a requirement

Believe it or not, the Driver interface and associated driver classes are not critical infrastructure for OpenMDAO. 
That sounds weird since drivers provide the "O" in MDAO, but weird or not it is true. 
The Driver interface is nice to have, because it handles a bunch of subtle details for you and provides a single generic interface that you can count on.
"Nice to have" is not the same thing as "critical infrastructure" though. 
Using just the Problem level APIs, you can absolutely write your own interface to you optimization library of choice. 

## Why have Drivers at all? 

We never even considered this question until RevHack2020, 
but thinking about the proposed CMA-ES problem it caused us to re-evaluate the very existence of drivers. 

### Spoiler: OpenMDAO is not going to remove the drivers

If for no other reason than the backwards compatibility hit would be too large, we are not going to get rid of drivers.
Id like to share the details we considered about the issue anyway, because they might help you make your own decision about whether you use them or not going forward. 

### Basic data on drivers: 
* There are about 8000 lines of code in the [OpenMDAO drivers module](https://github.com/OpenMDAO/OpenMDAO/tree/2186adb1ba66e0babaad8d2e6c7da071e1c6e973/openmdao/drivers)
* The code itself is about 1200 lines, and 6800 lines of that is for tests (are you surprised by that split? it takes a lot of effort to build a reliable test suite!)
* The [Driver base class](https://github.com/OpenMDAO/OpenMDAO/blob/2186adb1ba66e0babaad8d2e6c7da071e1c6e973/openmdao/core/driver.py) adds another 1200 lines of code (its located in the `core` module) and 1000 lines of test code. 
* As of V3.4, there are 5 drivers: [ScipyOptimizeDriver][scipy-driver], [pyOptSparseDriver][pyopt-driver], [SimpleGADriver][simple-ga], [DifferentialEvolutionDriver][di-ga], [DOEDriver][doe-driver]
* OpenMDAO devs developed an plugin (i.e. not in the main repo) experimental mixed integer optimizer called [AMIEGO](https://github.com/Kenneth-T-Moore/AMIEGO)
* National Renewable Energy Lab (NREL) built a plugin with a [driver for NL-Opt](https://github.com/johnjasa/nrel_openmdao_extensions)
* [Onera users built a plugin](https://github.com/OneraHub/openmdao_extensions) with several additional drivers for design of experiments and surrogate based optimization

### What does using a Driver give you that rolling your own run-script doesn't? 

At first glance, this is a somewhat surprisingly tough question to answer. 
For "simple" optimization problems --- simple from the perspective that they have a relatively small number of design variables and a small number of constraints --- the answer is not a whole lot! 
The one significant feature they offer is the handling of all the scaling (i.e. ref/ref0) and unit conversion between the model and the driver. 
This is pretty important, and surprisingly hard to deal with in the most general case. 
Another thing that they offer is integration with OpenMDAO's case recorder system, but depending on how you feel about our case recorders that could be either a positive or a negative. 

There are some useful features that come into play for more complex optimization problems formulations though: 
* separate handling of linear and nonlinear constraints (OpenMDAO drivers compute derivatives for linear constraints only once and cache the results)
* Efficient support for double-sided constraints (i.e. `0 < x < 10`). Here, we can compute one derivative that applies to both sides of the constraint with only a sign difference
* Tight integration with  [total derivative coloring](http://openmdao.org/twodocs/versions/3.4.0/features/core_features/working_with_derivatives/simul_derivs.html). 
* [Driver debug printing](http://openmdao.org/twodocs/versions/3.4.0/features/debugging/debugging_drivers.html)

Every feature, except the debug printing, is related to derivatives. 
If you're not using analytic derivatives then it seems like the value of the driver is pretty small, 
and it could be totally outweighed by some of the complexity it brings. 

### Why do drivers require so much code? 

The sample run-script for `<insert some optimizer library here>` is usually less than 100 lines of code. 
Drivers use way more code (the pyOptSparse driver is about 800 lines long) not even including all the tests. 
Why so much extra complexity? 

Mostly this comes down to the need for supporting more advanced use cases. 
Yes, for "simple" optimizations (again, referring the relative complexity of the optimization formulations)  
you can have very compact run-scripts. 
However, if you want to set up a big problem that has a sparse jacobian, uses MPI parallelism, has linear constraints, relies on derivative coloring schemes, or includes detailed data recording and debugging features then you'll find that your run script grows pretty quickly. 

I think it's easy to discount the slow creeping complexity growth of a run-script, especially if you're the only one working on that script. 
However, when ALL of the problems your solving are big, complex, optimizations that are changing all the time... the complexity of the drivers starts to make a lot more sense and add a lot more value. 

### Drivers make it easier to switch between optimization libraries ... sort of

One of the original motivations for drivers (going all the way back to V0) was to provide a unified interface so you could switch between Scipy, pyOptSparse, or whatever other driver you built yourself. 
In hindsight, I now think that goal was overly naive. 
There are lots of different optimization aggregation libraries, like pyOptSparse, NLOPT, and Dakota out there. 
Each takes a slightly different approach to their APIs that tailor toward one or another application and coding style. 
Adding OpenMDAO's Driver interface as another layer on top of that adds some value, but also adds another layer of code to dig through when debugging and wipes away some of the valid differences in API choices between libraries. 

For the dev team, having a unified interface between pyOptSparse and Scipy is pretty critical. 
We use pyOptSparse in our production problems because it couples to both SNOPT and IPOPT well, and has the key sparsity features we need to solve big optimizations. 
However, we know that not everyone has SNOPT or IPOPT installed and that (especially on windows) compiling pyOptSparse can sometimes be a huge obstacle. 

Scipy's optimizer library offers a great lowest-common-denominator for all our users. 
It doesn't have the best optimizers out there, but its not terrible either and everyone can easily get it pre-compiled. 
So we need to test with both, and hence we need to be able to switch between them with 0 effort. 
That means the dev team would at least need our own internal Driver-like tools for ourselves. 


## Drivers are not going away. 

Despite there being some problems with drivers, on balance they offer a lot of value. 
I wish there was a better way to offer the broad feature set and generality without all the complexity, but as of now I don't see one. 
The devs can't take on the responsibility of maintaining drivers for every optimization package, but we also don't think that you have to have a driver for your library of choice to use OpenMDAO. 
If you are willing to sacrifice the generality, then a lot of the Driver complexity isn't needed. 

The solution seems clear. 
Usage of the Driver interface should be optional. 
The devs should provide some good, clear examples of how to optimize things without drivers the docs. 
This needs to include links to all the relevant APIs for handling things like scaling, and derivative coloring. 

Our primary recommended usage pattern will still be to leverage drivers, specifically because they have the generality needed to support a wide range of problems. 
But we'll make it clear that there is another option as well. 





[CAM-ES]: ../problems/cma_es/README.md
[scipy-driver]: http://openmdao.org/twodocs/versions/3.4.0/features/building_blocks/drivers/scipy_optimize_driver.html
[pyopt-driver]: http://openmdao.org/twodocs/versions/3.4.0/features/building_blocks/drivers/pyoptsparse_driver.html
[simple-ga]: http://openmdao.org/twodocs/versions/3.4.0/features/building_blocks/drivers/genetic_algorithm.html
[di-ga]: http://openmdao.org/twodocs/versions/3.4.0/features/building_blocks/drivers/differential_evolution.html
[doe-driver]: http://openmdao.org/twodocs/versions/3.4.0/features/building_blocks/drivers/doe_driver.html

