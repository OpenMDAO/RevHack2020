#Using OpenMDAO as a compute engine, and writing your own run-scripts around it

## OpenMDAO doesn't have to be on top! 

It may be surprising to you, since we admittedly don't show you any examples of this in our docs, but this is an expected use for OpenMDAO. 
In fact, we've fairly recently added a [matrix-free total derivative feature][1] that lets you efficiently propagate analytic derivatives through an entire OpenMDAO model for situations where you want to build a for-loop around it, or tie it into some larger derivative framework. 
We'll make specific use of the matrix-free total derivatives when we implement a for-loop style time integration in a one of our problem solutions. 

## Here is an interesting thought experiment: Does OpenMDAO even need drivers? 

What if we didn't have drivers at all? 
Instead users would be responsible for linking their problems into the optimization library of their choice? 

If you like our Driver interface, don't worry; we are not getting rid of it. 
None the less, we could argue that the driver interface (and all of the optimizers that follow from it) are not critical to OpenMDAO (except that the "O" in the name wouldn't make much sense any more). 
In the hackathon, we'll show you an example of how this would work with the CMA-ES library. 
A number of users have asked about this, including [Dr. Barter's 2019 talk][2] also suggested that NL-opt would be useful to them. 


## Why have drivers at all? 
Driver's are useful, because the handle a lot of details about optimizer integration for you. 
For instance, they cache any linear derivatives so you only compute them once and they handle details about broadcasting design variables to all processes under MPI. 
Unfortunately, their generality also makes them complex and in some cases hard to debug. 
In our opinion, its 100% valid for you to not use Drivers at all (especailly if you don't need the more advanced features). 

So its reasonable to feel a bit overwhelmed when you look at driver code and wonder why it needs to be so hard to link your favorite optimizer in with OpenMDAO. 
As you'll see, it absolutely doesn't have to be! 

## A simple example with Scipy
Here is a really quick example of using [scipy's built in optimizer][3] as the top level optimizer around the [sellar problem][4] form the OpenMDAO tutorials. 


## Hackathon solutions that use this solution approach 
* Build an unsteady VLM simulation using an OpenAeroStruct model as a base
* Optimize an OpenMDAO model with CMA-ES optimizer


## Important OpenMDAO APIs for this solution approach 
* run_model 
* compute_totals 
* matrix-free total derivatives


[1]: http://openmdao.org/twodocs/versions/3.4.0/features/core_features/working_with_derivatives/total_compute_jacvec_product.html
[2]: https://www.youtube.com/watch?v=OlL1QmtLQQw&list=PLPusXFXT29sXIwZfZf3tLs3wr1sPk7d5J&index=6
[3]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
[4]: http://openmdao.org/twodocs/versions/3.4.0/basic_guide/first_mdao.html