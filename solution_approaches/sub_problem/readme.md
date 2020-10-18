# Using sub-problems in OpenMDAO

This topic has come up a a bunch of times over the years. 
The most common use case is for nested optimization, either for formal 
MDO architectures or more custom one-off solutions. 
But, as you'll see there are other good use cases for it too. 

## Nested Optimization can be done with OpenMDAO V3! 
I know ... you don't believe me.
That is understandable.  

Nested optimization has been a bit of a hot-topic since the transition from V0 to V1. 
OpenMDAO V0 explicitly supported it -- we wrote a [paper benchmarking MDO architectures][9].
However, the V1 rewrite dropped the explicit support for nested optimization, 
but you could still achieve it with sub-problems (a.k.a. problem within a problem). 
Toward the end of the V1 development, we actually implemented a generalized sub-problem feature. 
Then in the V2 re-write the sub-problem feature didn't get ported and still isn't in V3.4. 

Since we don't include the sub-problem in the main code base, it is understandable that you think we don't want them to use sub-problems for nested optimization. 
Even though it is a reasonable conclusion, it is only half right. 
True, in our opinion monolithic optimization is better approach for most situations. 
However, lots of very good work on nested optimization has shown it has value. 
Even if we don't choose it for our work, 
there is no reason that users can not or should not choose it for theirs. 

## Sub-problems have other uses too 
sub-problems are a useful abstraction when you want to roll your own custom time integration schemes. 
They can also be useful if you need to do something funky with a sub-part of your model within a larger model, such as finite-difference to include a derivative as a constraint. 

## If sub-problems are useful, why did you take them out in V2 and V3?  

They were dropped from V0 because that implementation wasn't really a sub-problem, as much as a special kind of opaque group (they were called Assemblies back then). 
These opaque groups created fundamental problems that were not fixable. 
I know that "opaque-group" vs. "sub-problem" does not give you a lot of details to work with, but suffice it to say that the former is a bad design and the latter a good one. 
In the rewrite from V0 to V1, so many were more pressing to re-implement than sub-problems. 
They stayed low on the priority list in large part because we didn't need to use them ourselves, but also because we felt they were fairly easy for users to implement on an ad-hoc basis as needed. 
Still we did eventually get to it, only to drop the feature again in V2. 

## you can roll your own sub-problems

The lack of support for them in V2 and V3 just boils down due to our perception that it is simple enough to implement them yourselves when you need them. 
However, the idea submissions for RevHack2020 have produced so many use cases where they would be valuable. 
So our only logical conclusion is that we need to, at the very least, show some good clear examples of how to implement them and express our support of them as a valid model building tool. 

Our end goal for RevHack isn't to produce a fully flushed out official sub-problem feature for OpenMDAO, but to demonstrate how to make them when you need them. 
If we come up with a nice generalized solution, we'll probably release it as a plugin first. 
Depending on its use, we'll consider adding it as to the main repo in the future. 


## A simple example of time-stepping with euler integration and sub-problems 

## Hackathon solutions that use this solution approach
* Use the analytic derivatives in OpenAeroStruct] to optimize an aircraft subject stability constraints 
* Demonstrate recommended nested optimization approach

## Important OpenMDAO APIs for this solution approach 
* run_model 
* add_design_var 
* add_objective 
* add_constraint 
* compute_totals 
* matrix-free total derivatives

[9]: http://openmdao.org/pubs/Gray_Moore_Hearn_Naylor-_2013_-Benchmarking.pdf
