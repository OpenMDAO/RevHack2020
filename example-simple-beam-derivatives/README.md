# OpenMDAO component for a cantilever beam

This problem is fairly simple from an engineering standpoint, 
but the sparsity pattern for the analytic derivatives is moderately complex. 
The code provided implements the nonlinear solver, but no derivatives are given. 

## Request: 
1) Add analytic derivatives using sparse partial derivatives and describe the process of determining the sparsity pattern.
2) Compare compute cost computing total derivatives with FD partials, 
CS with partial coloring, and sparse analytic partials

