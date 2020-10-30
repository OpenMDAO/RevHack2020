# OpenMDAO Driver for CMA-ES optimizer

The CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is an evolutionary algorithm for difficult non-linear non-convex black-box optimisation problems in continuous domain... 

* [INRIA reference site](http://cma.gforge.inria.fr)

* [Python source code](https://github.com/CMA-ES/pycma)  

* [API entry point](http://cma.gforge.inria.fr/apidocs-pycma/cma.html)

## Request:
Initiate an OpenMDAO driver for the CMA-ES optimizer.

As an evolutionary algorithm, the driver should be something along the lines of the SimpleGADriver. 
We want to be able to run in parallel as well. 

Would it be interesting to subclass or identify some common base class for evolutionary optimizers yet? 

## Solutions:

1. Custom run script using the native CMA-ES optimizer

    This approach involves wrapping an OpenMDAO Component or model with a function, so that it the CMAES library can be used as intended via it's functional interface. See [this Jupyter Notebook](cmaes.ipynb) for an example of how this is accomplished.

2. OpenMDAO Driver using CMA-ES optimizer

    As an initial step, an OpenMDAO [CMAESDriver](cmaes_driver.py) class was implemented based heavily on the existing [DifferentialEvolutionDriver](https://github.com/OpenMDAO/OpenMDAO/blob/master/openmdao/drivers/differential_evolution_driver.py).   See [this Jupyter Notebook](CMAESDriver.ipynb) to see the driver in action.

    This implementation uses the simple [functional](http://cma.gforge.inria.fr/apidocs-pycma/cma.evolution_strategy.html#fmin) `cmaes` API when running in serial, and the [object-oriented](http://cma.gforge.inria.fr/apidocs-pycma/cma.interfaces.OOOptimizer.html) API for parallel execution. The object-oriented API is useful because it allows for custom logic after case generation.  For parallel execution, we are able to use OpenMDAO's infrastructure for concurrent execution to evaluate the generated cases using all available processors.

3. Common base class for evolutionary optimizers

    In adapting the DifferentialEvolutionDriver to use the `cmaes` library, it was clear that there was a significant amount of common driver code that could be factored out such that different algorithms could more easily be implemented as an OpenMDAO driver.  A [GenericDriver](generic_driver.py) base class was implemented with the common code that was factored out of the existing drivers.  This driver uses a `DriverAlgorithm` object that contains the algorithm-specific logic. A rough re-implementation of both the Differential Evolution and CMAES drivers was done based on this generic driver base class (see [test_generic_devol](test/test_generic_devol_driver.py) and [test_generic_cmaes](test/test_generic_cmaes_driver.py)).

    More work is necessary to shake out bugs and corner cases, but the supposition in the proposal is shown to be correct.  OpenMDAO's [SimpleGADriver](https://github.com/OpenMDAO/OpenMDAO/blob/master/openmdao/drivers/genetic_algorithm_driver.py) is also a candidate for refactoring to use the base class, and may clarify additional requirements.
