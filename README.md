
# OpenMDAO RevHack 2020

## Watch this repo to get notifications as we push updates throughout the day during the two weeks of RevHack2020

RevHack is a reverse hackathon where the OM users propose problems to be solved in OpenMDAO and then the development team tries to solve them! 
The motivation for the dev team is to gain a better understanding of the use cases that our users are struggling with. 

The dev team goal for this hackathon was to get a better idea of the kinds of challenges our users were struggling with. 
We noticed three general solution approaches that collectively would span the entire set of problems submitted.

## Common solution approaches that will be useful to users
* Using OpenMDAO as a compute engine, and writing your own run-scripts around it
* Using sub-problems in OpenMDAO
* Building unsteady/transient analysis in OpenMDAO

## Why aren't these solution approaches in the docs already? 
Good question! 
The ability to use OpenMDAO these ways is part of the design intent of the framework, 
which is why we already had the features in place to quickly implement each of these approaches. 
However, this way of executing things isn't the way that the NASA users typically do things,
so we simply didn't think to document them. 

Though we have not typically run OpenMDAO this way in-house, 
these are still perfectly good ways of using OpenMDAO! 
In fact, we will probably adopt at least some of these things into our common use going forward. 
We are grateful to the efforts of those who submitted projects for helping us see alternate ways. 


## Is there an "OpenMDAO Way"?
OpenMDAO V3 has introduced a lot of API refinement and performance gains to the platform. 
One effect of these improvements is a new usage of OpenMDAO as low-level computational layer for building new generalized modeling libraries. 
There are some notable successes that heavily exploit the analytic derivative features in OpenMDAO: [OpenAeroStruct][1] (3 different revhack submissions include it!), [pyCycle][4], [Dymos][2], [OpenConcept][5].
These codes all adopt coding style with OpenMDAO as the top level execution manager and a single level monolithic optimization. 
Is this the "OpenMDAO way"? 
It is at least one way, but it's certainly not the only way. 

There are definitely times when it makes sense to have something else (i.e. not OpenMDAO) be the top level execution manager. 
Not every problem is most effectively solved as a single monolithic block with a single optimizer. 
So we'll show you how we'd implement these thins using OpenMDAO as a part of a larger whole. 

Moving forward we'd like to have OpenMDAO be seen as one tool in your tool box, instead of some kind of super-multi-tool that should be the only one you need! 


# The OpenMDAO RevHack 2020 Development Plan

There were 8 ideas submitted. 
We're rejecting 2 and accepting the remaining 6. 
Throughout the next two weeks we'll develop some general tools to tackle these 6 problems, 
then use them to implement specific solutions for each problem. 
For a few of the problems, we think there are multiple good ways to tackle it and (time permitting) we'll implement multiple solutions. 

## 6 accepted ideas
* Build an unsteady VLM simulation using an [OpenAeroStruct][1] model as a base (@shamsheersc19)
* Integrate an [OpenAeroStruct][1] analysis into a [Dymos][2] trajectory analysis model (@shamsheersc19)
* Use the analytic derivatives in [OpenAeroStruct][1] to optimize an aircraft subject stability constraints (achase90)
* Create a [Dymos][2] implementation of an eVTOL takeoff optimization that already exists as a explicit time integration implementation in OpenMDAO (@shamsheersc19)
* Demonstrate recommended nested optimization approach (@johnjasa)
* Optimize an OpenMDAO model with [CMA-ES][10] optimizer (@relf)

## 2 rejected ideas
* ~~Write a new solver that provides nonlinear preconditioning (@anilyil)~~
* ~~Create a recommended installation procedure for researchers who want access to OM source code (@shamsheersc19)~~

These two are still awesome ideas, 
but we deemed them to be outside the scope for the reverse hackathon.
We'll still take a look at them, but not during this activity. 









[0]: https://openmdao.org/2020-openmdao-reverse-hackathon/
[1]: https://github.com/mdolab/OpenAeroStruct
[2]: https://github.com/OpenMDAO/dymos
[3]: https://www.youtube.com/watch?v=OlL1QmtLQQw&list=PLPusXFXT29sXIwZfZf3tLs3wr1sPk7d5J&index=6
[4]: https://github.com/OpenMDAO/pyCycle
[5]: https://github.com/mdolab/openconcept
[6]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
[7]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
[8]: http://openmdao.org/twodocs/versions/3.4.0/features/core_features/working_with_derivatives/total_compute_jacvec_product.html
[10]: https://github.com/CMA-ES/pycma