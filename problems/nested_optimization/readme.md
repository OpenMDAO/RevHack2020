# Nested optimization

NREL uses OpenMDAO for multiple tools and studies, including [WISDEM](https://github.com/WISDEM/WISDEM/).
We often want to formulate complicated multidisciplinary optimization problems in a modular way.
This might involve a top-level optimizer with internal sub-optimizations or sub-solvers due to the heterogeneous tools and complex systems being modeled.
The use of nested optimization occurs in WISDEM in a few different ways, but we are not sure of the best practices for setting up these problems within OpenMDAO.
We ask that the OpenMDAO team investigate and detail how best to set up and solve nested optimization problems of a few different types.
We understand that OpenMDAO isn't designed to handle nested optimization cases natively, but we encounter these cases often.

## Examples of when we use nested optimization

Nested optimization occurs within our wind energy problems in a few different ways.
Here is a non-exhaustive list of cases:

1.	Need to optimize pitch settings at all operating points (akin to optimizing trim settings on an aircraft). Currently we just use many calls to the Scipy optimizers in a Component.
2.	Need to optimize both aero & structural designs, but there is an imbalance in the computational time for each set of DVs. Currently we either do a monolithic optimization or manually break it up into two separate design problems.
3.	Multifidelity design optimization where there is an outer-loop optimization driving a series of inner loop optimizations. Currently don’t see how to do this in OpenMDAO, so we're using Scipy optimizers wrapped in a homemade top-level optimizer.

## Example simplified code

[Actual use cases within WISDEM that involve nested optimization are quite complicated](https://github.com/WISDEM/WISDEM/blob/b4563abea6e739930c056d85204f2037a41ea3d5/wisdem/servose/servose.py#L184), so we have developed a simple set of components that can represent a few different cases.
The physics and equations represented in these components are completely fabricated and are not representations of the exact problems we're trying to solve.
We wanted to make them simple enough to quickly understand.

First, I'll introduce the building blocks for this simple set of code, then I'll explain the run scripts that use those blocks.

`compute_pitch_angles.py` and `design_airfoil.py` both contain a single component very loosely representative of those we find in WISDEM.
`ComputePitchAngles` is an ExplicitComponent that has multiple internal calls to Scipy's `minimize()` function to solve for the best pitch angles for a wind turbine's blades at different wind speeds.
It also computes the power produced by the wind turbine at each of the wind speeds.
`DesignAirfoil` is a simple ExplicitComponent that computes some measure of efficiency based on an `airfoil_design` design variable.
Lastly, `compute_modified_power.py` contains an ExplicitComponent that computes the `modified_power` based on the powers and efficiencies outputted from the other two components.

Then, we have three optimization scripts that use these components.

* `run_pitch_angle_opt.py` performs an optimization using only the single component `ComputePitchAngles`. Because this top-level optimization has nested calls to Scipy's minimize, it is a multilevel optimization problem.
* `run_MDF_opt.py` takes all three components and optimizes the combined metric `modified_power`, again using the nested calls to Scipy's minimize.
* `run_sequential_opt.py` sequentially optimizes subsets of these components, passing the optimal answer from one discipline to the other repeatedly. This is akin to doing aerostructural sequential optimization by passing the loads and displacements sequentially between the disciplines.

## Requests

* Please transform these provided examples by using the best practices for nested optimization in OpenMDAO.
* Discuss if we had a problem with a different model architecture or optimization formulation how would we best use nested optimization.
* Comment a bit on how to use these best practices in larger, more complex cases, like within WISDEM.
* Discuss if native nested optimization support is in OpenMDAO's pipeline and the associated decisions behind that development.

