# Implementing nonlinear preconditioning methods in OpenMDAO

## Background

Similar to linear preconditioning, nonlinear preconditioning methods try to change the nonlinear system to a different nonlinear system that has the same solution as the original system, but is easier to solve.
It is well known for the Newton's method to converge rapidly once the state is close to the final solution; however, far away from the solution, the method suffers from slow convergence, or may even stall due to local minima.
This is caused by multiple mechanisms, but the mechanism we focus on resolving is due to unbalanced nonlinearities, where a subset of the domain is slowing down convergence in the whole system due to the increased nonlinearity it has compared to the rest of the domain.
The main idea in using nonlinear preconditioning in this context is to balance these imbalanced nonlinearities to accelerate Newton's method even when the initial guess is not near the final solution.

The family of nonlinear preconditioning methods we are interested in for proposal are based on domain decomposition; specifically, they are based on the additive or multiplicative Schwarz methods.
Due to the "component" approach in OpenMDAO, there is a natural way to decompose the solution into subdomains, where the smallest subdomain is a component, but we can also consider multiple components together for a single subdomain.

The original additive Schwarz preconditioned inexact Newton (ASPIN) method [1] was introduced as a nonlinear preconditioning method for parallel CFD simulations, where the parallel domain decomposition provides a natural application to the domain decomposition used for nonlinear preconditioning.
With this method, the authors use the additive Schwarz method with overlap to build a preconditioned nonlinear system, and use an inexact Jacobian to solve the resulting system with the Newton's method, hence the name "inexact Newton". 

While the additive Schwarz version works well for parallel computations, where we prefer to do as many computations as possible with minimal communication, for a typical OpenMDAO model, there are multiple low-cost components that perform the computations in serial. Due to this property, the multiplicative Schwarz variant is more useful, where updated states from previous components are used for the next component.
In this context, additive Schwarz can be seen as a block-Jacobi approach, where multiplicative Schwarz is a block-Gauss--Seidel approach.

The multiplicative Schwarz preconditioned inexact Newton (MSPIN) method was introduced in [2], and this approach is the preferred method for serial computations and is expected to yield better convergence rates, as long as the computations are not overwhelmed by parallel communication.
In this paper, the authors again used an inexact Jacobian to solve the resulting nonlinear system; however, compared to the ASPIN paper, this paper does not include any overlap between the sub-domains.
Since the models we are interested in for now are purely serial models, the MSPIN approach seems like a natural choice for us.

Finally, the hierarchical Newton method currently used in OpenMDAO can also be formulated as a nonlinear preconditioning method similar to these; however, there a subtle differences in the final nonlinear system. 
In my notes, I have shown that the hierarchical Newton method approaches the multiplicative Schwarz based nonlinear preconditioning as we approach the solution; however, the differences between the methods can be large when we are far away from the solution, and this is the main part of the nonlinear convergence we want to accelerate.
Therefore, I see a benefit in at least trying the additive and multiplicative Schwarz based nonlinear preconditioning in OpenMDAO models.

## Request

In this proposal, I request that the OpenMDAO development team try implement two nonlinear preconditioning methods: additive Schwarz preconditioned "exact" Newton (ASPEN) and multiplicative Schwarz preconditioned "exact" Newton (MSPEN), without any overlap for either method for simplicity.
In both implementations, instead of the inexact Jacobian matrices used in [1,2], we can actually use the exact Jacobian formulation.
This is primarily because for now we are interested in simple test cases, and we are also focusing on an all matrix-based implementation, rather than a matrix-free Newton formulation.
Finally, having no overlap in the Schwarz methods greatly simplify the analytic Jacobian formulation. 

The hierarchical Newton machinery already implemented in OpenMDAO will be useful when building these methods, but I am not sure how the actual software implementation would work. 
One can either implement these methods as separate nonlinear solvers, or as a new class of nonlinear preconditioning methods that are used with existing nonlinear solvers.
The former will most likely result in a more efficient implementation, while the latter will be much more flexible, since both additive and multiplicative Schwarz methods can be used without the overarching Newton's method, or with other nonlinear solvers.

This proposal is different than other reverse-hackathon proposals because it is focused on changes to OpenMDAO itself, rather than building models with it.
However, I believe these nonlinear preconditioning methods have the potential to evolve into much more robust and adaptive nonlinear solver strategies, where they can tackle models that are notoriously difficult to converge (see most models that use pyCycle).

## Algorithms

Section 3.1 of [2] details both ASPIN and MSPIN algorithms.
The Jacobian they use for the overarching Newton solver is not the exact Jacobian, but we can use an exact Jacobian with minor modifications to what the use.
In particular, the differences are the following for each method:

ASPIN to ASPEN: The Jacobian in ASPIN contains the partials of G and H, evaluated at u and v, which are the initial guesses going into this iteration. However, an exact formulation would require differentiating all the G terms at the states p and v, and also differentiating all H terms at u and q.
See equation 2.12 for the Jacobian of ASPEN, and 2.13 for the Jacobian of ASPIN in reference [2].

MSPIN to MSPEN: Similar to the additive version, the G terms should be differentiated at p and v.
However, with the multiplicative version, since we use the updated state p to evaluate the H residual, we need to differentiate the H terms at p and q. 
See equation 2.28 for the Jacobian of MSPEN, and 2.29 for the Jacobian of MSPIN in reference [2].

While both of these derivations in the paper are done for a 2-component system, they easily extend to multiple components.
In the appendix of [2], the authors present the Jacobian for an N-component MSPIN method, and the N-component ASPIN Jacobian is easy to formulate. 
Similarly, we can modify these Jacobian formulations to get the respective N-component ASPEN and MSPEN variants.

The notation used in [1] and [2] is maybe cleaner and easier to understand, but I do not like this notation. 
A better notation is used in [3]; however, they also changed the preconditioned nonlinear residual, and the derivation is only done for the additive version, and not the multiplicative version, though this is easy to extend.
In [3], g and h refer to the solution operators defined on the subdomains directly, whereas in [1] and [2], g and h terms are the negative deltas to the baseline solution u and v to solve the subdomain systems.

Finally, because we do not use any overlap for now, we do not need to specify how the overlap is formulated.
The plain additive Schwarz method simply adds the contributions from both subdomains to the global solution, resulting in a non-convergent scheme if it is used without a Newton's method over the whole system.
An improved restricted additive Schwarz (RAS) method restricts the update vector to the subdomain itself without overlap, and as a result, no double-counting of updates happen.
Unlike plain additive Schwarz, RAS actually provides a convergent iteration, and it is the formulation used in [3]. 
However, since we do not have any overlap in these simple examples in the first place, we do not need to worry about this aspect.

## Test problems

The simplest test case for these methods is the 2-variable nonlinear system introduced in Section 3.2 in [2].
Using this example, the authors formulated the residual contours to visualize how the nonlinear preconditioning turns a complex nonlinear surface into an elliptical one, where the Newton's method is expected to perform better.
One basic check is that the both MSPEN and ASPEN residuals do not get affected by the inexact Jacobian, and therefore, to validate our implementation, we can try to recreate the same plots as in the paper.
Furthermore, I have modified the hierarchical Newton to have a similar notation in my notes, so that we can also plot the resulting preconditioned nonlinear system contours and compare different approaches.
Both MSPEN and ASPEN require a good amount of extra computations than hierarchical Newton, and therefore we expect the residual contours to be considerably better than the contours from hierarchical Newton for them to make sense in terms of performance.

Besides this simple test, I plan on cleaning up the scalable test problem developed by Sham in [4].
Finally, we can try the method on a difficult pyCycle or OpenConcept model.
I will provide the OpenMDAO models for all of these tests, although the 2-component test case is extremely easy to code, so feel free to create your implementation of it.

## References

[1] Cai, X.-C., and Keyes, D. E., “Nonlinearly Preconditioned Inexact Newton Algorithms”, SIAM Journal on Scientific Computing, Vol. 24, No. 1, 2002, pp. 183–200. 
doi: [10.1137/S106482750037620X](http://dx.doi.org/10.1137/S106482750037620X)

[2] Liu, L., and Keyes, D. E., “Field-Split Preconditioned Inexact Newton Algorithms”, SIAM Journal on Scientific Computing, Vol. 37, No. 3, 2015, pp. A1388–A1409. 
doi: [10.1137/140970379](http://dx.doi.org/10.1137/140970379) 

[3] Dolean, V., Gander, M.J., Kheriji, W., Kwok, F., and Masson, R., “Nonlinear Preconditioning: How to Use a Nonlinear Schwarz Method to Precondition Newton’s Method”, SIAM Journal on Scientific Computing, Vol. 38, No. 6, 2016, pp. A3357–A3380. 
doi: [10.1137/15M102887X](http://dx.doi.org/10.1137/15M102887X)

[4] Chauhan, S. S., Hwang, J. T., and Martins, J. R. R. A., “An automated selection algorithm for nonlinear solvers in MDO”, Structural and Multidisciplinary Optimization, Vol. 58, No. 2, 2018, pp. 349–377. 
doi: [10.1007/s00158-018-2004-5](http://dx.doi.org/10.1007/s00158-018-2004-5)