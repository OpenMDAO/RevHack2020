# OpenMDAO RevHack 2020

This is the shared repository for OpenMDAO RevHack 2020. 

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
There are some notable successes that heavily exploit the analytic derivative features in OpenMDAO: [OpenAeroStruct][1] (3 different revhack submissions include it!), [pyCycle][4], [Dymos][1], [OpenConcept][5].
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


### Building unsteady/transient analysis in OpenMDAO: 
The eVTOL, OAS+Mission, and unsteady VLM ideas all deal with some kind of unsteady analysis. 
We see two ways to implement this kind of thing in OpenMDAO: 
* All times in memory at once (e.g. dymos, open-concept, sham's model?)
* time-stepping approaches with just one time-step in memory at once (i.e. for loops)

If you want to include analytic derivatives, the all times at once approach is the simplest to differentiate. 
It is also the least similar to how you would probably code up an unsteady analysis. 
You're more likely to use a hand coded for-loop, or scipy's [odeint][6] or [solve_ivp][7] features. 

*We are going to show you both ways and compare them. 
We will also provide our opinion for when you should choose one way or the other* 


### Using sub-problems in OpenMDAO
This topic has come up a a bunch of times over the years. 
One of the primary motivations for it seems to be nested optimization, and we got one problem idea to look at that exact topic. 
Some nested optimizations come in the form of formal MDO architectures, but others are more ad-hoc. 

Nested optimization has been a bit of a hot-topic since the transition from V0 to V1. 
OpenMDAO V0 explicitly supported it -- we wrote a [paper benchmarking MDO architectures][9].
However, the V1 rewrite dropped the explicit support for nested optimization, 
but you could still achieve it with sub-problems (a.k.a. problem within a problem). 
Toward the end of the V1 development, we actually implemented a generalized sub-problem feature. 
Then in the V2 re-write the sub-problem feature didn't get ported and still isn't in V3 either. 
This seems to have given a lot of users the impression that we don't want them to use sub-problems (or at least not to use them for nested optimization)
It is a reasonable conclusion to draw --- since we kept dropping sub-problem --- but it is not 100% accurate. 
True, in our opinion monolithic optimization is better approach for most situations. 
However, lots of very good work on nested optimization (a primary use case for sub-problems) has shown it has value. 
Even if we don't choose it on the dev team, there is no reason that users can not or should not choose it. 

Also, nested optimization isn't the only valuable use-case for sub-problems. 
There are some very good arguments for sub-problems in normal model building situations (we will show several of them as solutions to RevHack2020 problems). 

So if they are useful, why does the dev team keep dropping support for them? 
They were dropped from V0 because that implementation wasn't really a sub-problem, as much as a special kind of opaque group (they were called Assemblies back then). 
These opaque groups created fundamental problems that were not fixable. 
I know that "opaque-group" vs. "sub-problem" does not give you a lot of details to work with, but suffice it to say that the former is a bad design and the latter a good one. 
In the rewrite from V0 to V1, so many were more pressing to re-implement than sub-problems. 
They stayed low on the priority list in large part because we didn't need to use them ourselves, but also because we felt they were fairly easy for users to implement on an ad-hoc basis as needed. 
Still we did eventually get to it, only to drop the feature again in V2. 

The lack of support for them in V2 and V3 just boils down due to our perception that it is simple enough to implement them yourselves when you need them. 
However, the idea submissions for RevHack2020 have produced so many use cases where they would be valuable. 
So our only logical conclusion is that we need to, at the very least, show some good clear examples of how to implement them and express our support of them as a valid model building tool. 

*We will use sub-problems to build solutions for the nested optimization, stability derivative constranied optimization, and transient model building ideas.* 

Our end goal for RevHack isn't to produce a fully flushed out official sub-problem feature for OpenMDAO, but to demonstrate how to make them when you need them. 
If we come up with a nice generalized solution, we'll probably release it as a plugin first. 
Depending on its use, we'll consider adding it as to the main repo in the future. 



[0]: https://openmdao.org/2020-openmdao-reverse-hackathon/
[1]: https://github.com/mdolab/OpenAeroStruct
[2]: https://github.com/OpenMDAO/dymos
[3]: https://www.youtube.com/watch?v=OlL1QmtLQQw&list=PLPusXFXT29sXIwZfZf3tLs3wr1sPk7d5J&index=6
[4]: https://github.com/OpenMDAO/pyCycle
[5]: https://github.com/mdolab/openconcept
[6]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
[7]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
[8]: http://openmdao.org/twodocs/versions/3.4.0/features/core_features/working_with_derivatives/total_compute_jacvec_product.html
[9]: http://openmdao.org/pubs/Gray_Moore_Hearn_Naylor-_2013_-Benchmarking.pdf
[10]: https://github.com/CMA-ES/pycma