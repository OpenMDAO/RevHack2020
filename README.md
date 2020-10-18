# OpenMDAO RevHack 2020

This is the shared repository for OpenMDAO RevHack 2020. 

RevHack is a reverse hackathon where the OM users propose problems to be solved in OpenMDAO and then the development team tries to solve them! 
The motivation for the dev team is to gain a better understanding of the use cases that our users are struggling with. 

Special thanks go to @shamsheersc19 for submitting half of the total ideas, and more importantly for taking the time off-line to talk through some of the details with the one of the dev team members. 
As a long time user of OpenMDAO with a wide range of experiences using it in the classroom, in his own research, and advising others on how to do the same, Shamsheer gave us some important feedback about some serious usability issues that the community faces when adopting OpenMDAO. 
Shamsheer's insights gave us some new perspective on feature development and general product development strategy, and in particular where there are some holes. 

The original stated goal for the hackathon was to encourage adoption of the advanced features in OpenMDAO (e.g. analytic derivatives or MPI parallelism).
However, when we filtered the suggested problems through a usability lens it became clear that many users are wrestling with a different class of problems. 
In retrospect, Dr. Garett Barter's talk ["Growing Pains with OpenMDAO"][3], from the 2019 OpenMDAO workshop brought up a lot of the same issues. 


## Dev Team Current 2020 Retrospective on OpenMDAO
OpenMDAO V3 has introduced a lot of refinement and performance gains to the core platform. 
These important things polished the framework into an effective computational platform for building new models and generalized modeling libraries on top of. 
There are some notable successes that heavily exploit the analytic derivative features in OpenMDAO: [OpenAeroStruct][1] (3 different revhack submissions include it!), [pyCycle][4], [Dymos][1], [OpenConcept][5].
These codes all adopt a new coding paradigm that OpenMDAO introduces, in order to exploit these features to great effect.

However, many users are are struggling to fit their applications into this new coding paradigm. 
Things like nested optimization, greater optimizer variety, and wrapper scripts with for-loops and flexible logic are tougher fits. 
Broadly, there seems to be usability problem that stems from a combination of OpenMDAO's new coding paradigm, 
and users being unsure how to map their desired techniques to it. 

Without a doubt, OpenMDAO is highly tailored applications with analytic derivatives. 
That is not going to change. 
We still think that you should strongly consider using them. 
However, it shouldn't be a binary choice: OpenMDAO way or the old way. 
So there is a lot of value in helping users see clearer ways to map their desired approaches into OpenMDAO. 

## The NEW goal for OpenDMAO RevHack2020
To demonstrate some different approaches to using OpenMDAO as a part of a larger, more customized run time environment. 
We want to show you a different perspective where OpenMDAO is more of a low-level compute engine that you build on top of. 
We want to present an alternative to viewing OpenMDAO as the top of the stack! 


# The OpenMDAO RevHack 2020 Development Plan

## We've rejected 2 ideas
* ~~Write a new solver that provides nonlinear preconditioning (@anilyil)~~
* ~~Create a recommended installation procedure for researchers who want access to OM source code (@shamsheersc19)~~

These two are still awesome ideas, 
but we deemed them to be outside the scope for the new goals of the reverse hackathon.
We'll still take a look at them, but not during this activity. 

## We've accepted 6 ideas
* Build an unsteady VLM simulation using an [OpenAeroStruct][1] model as a base (@shamsheersc19)
* Integrate an [OpenAeroStruct][1] analysis into a [Dymos][2] trajectory analysis model (@shamsheersc19)
* Use the analytic derivatives in [OpenAeroStruct][1] to optimize an aircraft subject stability constraints (achase90)
* Create a [Dymos][2] implementation of an eVTOL takeoff optimization that already exists as a explicit time integration implementation in OpenMDAO (@shamsheersc19)
* Demonstrate recommended nested optimization approach (@johnjasa)
* Write a new driver that uses the CMA-ES optimizer (@relf)

There are some common themes that emerge when you look at all 6 together. 
We'll present them here as separate themes, but really they all overlap. 
When we get down to implementing solutions to these 6 ideas, we'll take parts from all these themes. 
You should look into the folder associated with each theme for a more in depth discussion of it, and clear links to where it applies to various solution. 



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

### Using OpenMDAO as a compute engine, and writing your own run-scripts around it
OpenMDAO doesn't have to be on top! 
It may be surprising to you, since we admittedly don't show you any examples of this in our docs, but this is actually an expected use for OpenMDAO. 
In fact, we've even added a [matrix-free total derivative feature][8] that lets you efficiently propagate analytic derivatives through an entire OpenMDAO model for situations where you want to build a for-loop around it, or tie it into some larger derivative framework. 
We'll make specific use of the matrix-free total derivatives when we implement a for-loop style time integration. 

Here is an interesting thought experiment: Does OpenMDAO even need drivers? What if we didn't have them at all, and users were responsible to linking their problems into the optimization library of their choice? 
If you like our Driver interface, don't worry; we are not getting rid of it. 
None the less, its interesting to realize that the driver interface (and all of the optimizers that follow from it) are not critical to OpenMDAO (except that the "O" in the name wouldn't make much sense any more). 
This comes into play in the CMA-ES idea for RevHack2020, but [Dr. Barter's 2019 talk][3] also suggested that NL-opt would be useful to them. 

Driver's are useful, because the handle a lot of details about optimizer integration for you. 
For instance, they cache any linear derivatives so you only compute them once and they handle details about broadcasting design variables to all processes under MPI. 
Unfortunately, their generality also makes them complex and in some cases hard to debug. 
In our opinion, its 100% valid for you to not use Drivers at all (especailly if you don't need the more advanced features). 

*We show you run script that wraps a stand-alone OpenMDAO model into the CMA-ES native interface.*

We'd love you input on if you prefer this way of doing things over leaving OpenMDAO on top. 

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




[1]: https://github.com/mdolab/OpenAeroStruct
[2]: https://github.com/OpenMDAO/dymos
[3]: https://www.youtube.com/watch?v=OlL1QmtLQQw&list=PLPusXFXT29sXIwZfZf3tLs3wr1sPk7d5J&index=6
[4]: https://github.com/OpenMDAO/pyCycle
[5]: https://github.com/mdolab/openconcept
[6]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
[7]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
[8]: http://openmdao.org/twodocs/versions/3.4.0/features/core_features/working_with_derivatives/total_compute_jacvec_product.html
[9]: http://openmdao.org/pubs/Gray_Moore_Hearn_Naylor-_2013_-Benchmarking.pdf