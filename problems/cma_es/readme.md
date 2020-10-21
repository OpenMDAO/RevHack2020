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

