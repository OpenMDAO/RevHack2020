# Using sub-problems in OpenMDAO

This topic has come up a a bunch of times over the years. 
The most common use case is for nested optimization, either for formal 
MDO architectures or more custom one-off solutions. 
But, as you'll see there are other good use cases for it too. 

## What is a sub-problem? 
A simple answer is this: a sub-problem is a `Problem` instance that gets embedded within the `model` of another OpenMDAO problem. 

The reality is just a little more complex. the `Problem` class in OpenMDAO does not support the same interface as the `Component` class. 
The `Component` interface is needed to be able to add something to a `model`, so in order to make sub-problems work you have to somehow add this interface to your sub-problem. 

The simple way to do this is to embed the sub-problem inside a component wrapper, like this: 

![cartoon diagram of a sub-problem](sub_problem_cartoon.png)

Sub-problems have lots of good use cases. 
The most commonly discussed one is for sub-optimizations. 
Another important one that came up in this hackathon is as a means to support time-stepping based unsteady analyses. 
A third use case also arose here, when sub-problems were valuable in allowing finite-differencing of a sub-model to use derivatives in a top level constraint on an optimization. 

## Nested Optimization can be done via Sub-Problems with V3! 
I know ... you don't believe me.
That is understandable.  

Nested optimization has been a bit of a hot-topic in MDO since the founding of the field. 
It has also been a contentious topic in the OpenMDAO user community since the transition from V0 to V1. 
OpenMDAO V0 explicitly supported it -- we wrote a [paper benchmarking MDO architectures][9].
The V1 rewrite dropped the explicit support for nested optimization
but you could still achieve it with sub-problems (a.k.a. problem within a problem). 
Toward the end of the V1 development, we actually implemented a generalized sub-problem feature. 
Then in the V2 re-write the sub-problem feature didn't get ported and still isn't in V3.4. 

Since we don't include the sub-problem feature in the main code base, 
and we no longer give any examples of sub-optimizations anywhere in the docs it is understandable to think we don't want you use them.
It is a reasonable conclusion, it is only half right. 
In our opinion monolithic optimization is better approach for most situations. 
There are good numerical arguments against sub-optimization (e.g. its really hard to get analytic derivatives across a sub optimization), 
but also many practical arguments against it too. 
For example, the SLSQP implementation is scipy is non-reentrant which means that you can really nest instances of minimize within itself and be sure of correct behavior. 

However, despite all the arguments that we could make against sub-optimizations, 
lots of very good work on nested optimization has shown it has value. 
Even if we don't choose it for our work, 
there is no reason that you can not choose it for yours! 

 
## If sub-problems are useful, why did you take them out in V2 and V3?  

They were dropped from V0 because that implementation wasn't really a sub-problem, as much as a special kind of opaque group (they were called Assemblies back then). 
These opaque groups created fundamental code structure problems that were not fixable. 
I know that "opaque-group" vs. "sub-problem" does not give you a lot of details to work with, but suffice it to say that the former is a bad design and the latter a good one. 
In the rewrite from V0 to V1, so many things were more pressing to re-implement than sub-problems. 
They stayed low on the priority list in large part because we didn't use them ourselves, 
but also because we felt it was pretty easy for users to implement ad-hoc versions as needed. 
Still we did eventually get to it, only to drop the feature again in V2. 

Why drop it again in V2? 
This time, really it was mostly an issue of priority.
We deeply value the input and feedback from our user community --- which is why we did RevHack2020 --- 
but we still have to prioritize the needs of our NASA ARMD sponsors above those of the outside users. 
Again, since the ad-hoc option was available, it just never made it to the top of the priority pile for us. 

## you can roll your own sub-problems

The dev team assumed that ad-hoc implementations of sub-problems were fairly strait forward, 
but the problem ideas we got submitted to RevHack 2020 universally didn't mention them or user them. 
So we've had revise re-evaluate our opinion on their strait forward-ness. 

Our conclusion is that we need to, at the very least, 
show some clear examples of how to implement them and express our support of them as a valid model building tool. 
We'll be adding some sections to the docs on this topic, but in the meantime here is a quick primer: 

## Important OpenMDAO APIs for building sub-problems
* `set_val` & `get_val`: useful methods for setting/getting variables that let you specify the units you want to work with. OpenMDAO then converts the values to the internal model units for you. This is helpful, because you don't ever need to look in a model to figure out what units it was built with. Check out the docs on this [here][http://openmdao.org/twodocs/versions/3.4.0/features/core_features/running/set_get.html#setting-and-getting-component-variables]
* `run_model()` & `run_driver()`: These are the methods that let you execute the model directly or execute the driver --- which will in turn execute the model iteratively for your. 
* Problem level case recording: You may not be aware of it, but you can [attach a case recorder to the problem itself][http://openmdao.org/twodocs/versions/3.4.0/features/recording/problem_options.html]. This is useful because it lets you manually trigger case recording whenever you want via the `record` method on the Problem. 
* `compute_totals`: this is a method on problem that gives you direct access to OpenMDAO's analytic derivatives engine. Check out the docs [here][http://openmdao.org/twodocs/versions/3.4.0/features/core_features/working_with_derivatives/compute_totals.html#computing-total-derivatives], which show you how to use it. 
`approx_totals`: This is a method on *Group* (note: not a method on Problem), but you may find it valuable none the less. 
You probably already knew that OpenMDAO could approximate partial derivatives of components, but you may not have known that it could [approximate semi-total derivatives][http://openmdao.org/twodocs/versions/3.4.0/features/core_features/working_with_derivatives/approximating_totals.html] across groups as well. This want to ask the problem to appoximate total derivatives across it, then you can call `approx_totals` on the `model` attribute of the problem to set this up. Then the `compute_totals` method will rely on that to give you total derivatives. 
* matrix-free total derivatives: If you want to efficiently propagate derivatives across a problem (e.g. for a time-integration for-loop) then this is a valuable feature. the [`compute_jacvec_product`][http://openmdao.org/twodocs/versions/3.4.0/features/core_features/working_with_derivatives/total_compute_jacvec_product.html] method on problem allows you to integrate a sub-problem into some higher level derivative calculation without having to assemble the full sub-problem total derivative jacobian. If you're trying to integrated a sub-problem into some external algorithmic differentiation system, this method will be critical for you. 

[9]: http://openmdao.org/pubs/Gray_Moore_Hearn_Naylor-_2013_-Benchmarking.pdf
