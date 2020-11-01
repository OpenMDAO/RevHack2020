# How big should I make my OpenMDAO components? 

Say you are developing a model in OpenMDAO and for the sake of narrowing the topic a bit, 
lets assume you are developing it all natively in python via a set of components ... 
so we are explicitly not worrying about wrapped codes or external libraries. 

It is common to ask how many chunks you should break it into? 
How many lines of code should you aim for? 
How do you know where to draw the boxes around the various parts of your calculation? 

If you used the tutorials in the docs as examples, 
you'd understandably get the impression that components should be pretty small. 
For example, [the Sellar problem][sellar] has two disciplines, each with their own component and each component with a single compute line. 
The [resistor and diode][circuit] components in the circuit example are pretty small too. 
Most of the [examples][examples] we show have pretty small components too. 

The size of components in the docs should not be considered an endorsement of smaller sized components. 
These are small specifically so that readers don't get bogged down in the details of a long code. 
The focus is on APIs, syntax, and clear readability. 

The "right" answer doesn't exist. 
OpenMDAO is flexible and can handle anything from tiny components to massive ones. 

## Example: pyCycle 

pyCycle --- a propulsion modeling library --- is built by members of the OpenMDAO devs. 
It tends to have much smaller components, though not exclusively so. Lets talk about why. 

## pyCycle was originally developed to be a stress test for OpenMDAO

Before the devs worked on OpenMDAO, many of us worked on [NASA's Numerical Propulsion Systems Simulation (NPSS)][npss] project. 
During the early development of the analytic derivative capability in V0, we needed something that would really stress that system. 
Given our familiarity with propulsion modeling and NPSS, it made sense to use that as a test case. 
By building pyCycle out of lots of little components, we tested two things. 
We made sure that OpenMDAO handled setup/compute efficiently when there were lots of components, 
and we made sure that the analytic derivative system was efficient in this situation too. 
It turned out we were right to be concerned, because pyCycle proved so challenging for OpenMDAO V0 that we had to drop that code base and start over with V1! 

So for this reason, you shouldn't consider pyCycle as a good reference for the right size of your components. 


## Smaller components are a lot easier to differentiate

That was the first code we wrote with derivatives, and it is **A LOT** easier to hand differentiate smaller components. 
That is another huge reasons why we kept them small. 
This reason is one you should also consider, especially if you are trying to train a group of new users. 

Manually differentiating components can prove to be one of the most expensive tasks in your development, 
so making smaller components can pay dividends by shrinking this job. 


## Many small components will (usually) be slower than fewer larger ones 

This is a bit hard to show with simple toy problems, which tend to have such insignificant compute operations that overhead swamps everything anyway. 
However, the devs have lots of examples where we've broken calculations up into smaller chunks and seen the code get slower. 

### Example: eVTOL trajectory optimization 
In fact, we have a great example from the eVTOL trajectory optimization solutions from this hackathon. 
We solved this problem by building an ordinary differential equation (ODE) of the problem dynamics, and then solving an trajectory optimization in our Dymos library. 
The general structure of the calculations was as follows: 



    +-----------------+
    |                 | +-------------+
    |  Pre-processing |               |
    |                 |     +---------v-------------+
    +-----------------+     |                       +-----------+
                            | Implicit calculation  |           |
                            |                       |           |
                            +-----------------------+   +-------v---------+
                                                        |                 |
                                                        | Post-processing |
                                                        +-----------------+




The diagram implies that 3 components would be appropriate so we tried that. 
We also grouped the first two chunks into a single component, and the made an ODE out of two components. 
We also tried grouping them all into one large component. 

Here is the optimization performance data: 

* 3 component ODE: 2:38 min 
* 2 component ODE: 2:41 min
* 1 component IDE: 1:35 min

So the 1 component ODE was about 2 times faster. 
It is also a lot easier to read the single component ODE code, since its all in one place. 
So win-win right? Yeah, in this case... because we used complex-step partials. 

If we had been hand differentiating these components, I would have gladly traded that run-time for less effort on the derivatives. 


### Counter Example: OpenAeroStruct (OAS) stability derivative optimization 

Of course, it wouldn't be a rule of thumb if there were not counter examples. 
We didn't have to look very far, since a different hackathon problem provided it! 

OpenAeroStruct (OAS) is a low-fidelity aerostructural solver, built natively in OpenMDAO. 
It tends to have many smaller components in order to make the derivatives easier to hand-derive. 
However (skipping a lot of details), when solving this problem we found that modest sized meshes were pretty slow. 
We used the OpenMDAO profiling tools to see that the cause of the problem was that there were a couple of components with outputs that got massive (i.e. the size of the arrays got very large) as the mesh grew and OpenMDAO's direct linear solver was bogging down a bit. 

So we decided to try and combine the components together, so the large arrays became intermediate variables that OpenMDAO didn't see. 
To make this work, hand differentiation wasn't an option any more. 
So we tried the algorithmic differentiation tool [JAX][jax], which worked surprisingly well. 
However, nothing is easy ... and in this case despite the rule-of-thumb, OAS was significantly slower with the larger super-component. 

In this case, the slowness was from JAX itself. 
We spent some time with their JIT, and improved things a bit, but the hand implementation with smaller components was still better. 
We need to do a bit more profiling at different mesh sizes, to see if there is now a cross-over point where the super-component wins out. 

The exception to the rule stands none the less. Here was a case where derivative computation costs dominated, and we were able to be more efficient by hand implementation despite requiring smaller components. 


## Recommendation: start-small to learn, but go big for production code 

We think that code is easier to read/debug if you make fewer larger components, and we would like to move more of our own code in that direction. 
So thats what we'll recommend to you as well. 
Tend toward larger components with more calculations aggregated into a single `compute` method. 

The major challenge with larger components it that they become much hard to provide derivatives for by hand. 
The solution to that problem is **Algorithmic differentiation (AD)**, but we hesitate to recommend you start out with AD from the get go. For one thing, AD support in python is ok but not as good as other languages. 
For another thing, you'll be a much better user of AD if you have some first hand experience with manual differentiation. 

For AD you have a few options. The first, is just to use the build in complex-step (CS) partial derivative approximation tools in OpenMDAO. 
CS is a form of forward-mode AD that is very easy to use, though you do need to be cautious because not all numpy functions are complex-safe. 
We've run into this enough that we've started a library of [CS-safe alternatives][cs-safe] in an OpenMDAO util package. 
If you are building components with inputs that are large vectors, CS can potentially get pretty expensive. 
To counter that, you can try our [partial derivative coloring features][partial-coloring] which may (or may not) help, depending on the sparsity patterns in your component. 

Another option is more traditional AD. 
We tried [JAX][jax] during this hackathon and had some good luck with it. 

In general, based on our experiences in this hackathon AD is something the devs are going to be investing our own time into more heavily. 
Its the key to making larger components work smoothly in OpenMDAO. 



[cs-safe]: https://github.com/OpenMDAO/OpenMDAO/blob/master/openmdao/utils/cs_safe.py
[partial-coloring]: http://openmdao.org/twodocs/versions/3.4.0/features/experimental/simul_coloring_fd_cs.html
[jax]: https://github.com/google/jax
[sellar]: http://openmdao.org/twodocs/versions/3.4.0/basic_guide/sellar.html#building-the-disciplinary-components
[circuit]: http://openmdao.org/twodocs/versions/3.4.0/advanced_guide/implicit_comps/defining_icomps.html#explicitcomponents-resistor-and-diode
[examples]: http://openmdao.org/twodocs/versions/3.4.0/examples/index.html
[npss]: https://software.nasa.gov/software/LEW-17051-1
[dp]: https://github.com/OpenMDAO/pyCycle/blob/d79e0b80305ba40fe03c24a0e5cb41dc857c520b/pycycle/elements/duct.py#L61-L82
[cea]: https://github.com/OpenMDAO/pyCycle/blob/d79e0b80305ba40fe03c24a0e5cb41dc857c520b/pycycle/thermo/cea/chem_eq.py#L45