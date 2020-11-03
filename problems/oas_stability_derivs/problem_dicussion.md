# Original Problem Statement

* We're interested in optimizing a vehicle subject to aerodynamic stability requirements. This is a
novel OpenMDAO problem because it requires constraining derivatives of components, not just
values themselves.

* We envision using OpenAeroStruct (since it has analytical derivatives) to drive parameters such as
dihedral and taper ratio to some objective (weight, range, etc.,) subject to stability derivatives
(Cnbeta, Cmalpha, etc.) being greater than some value.

* The aero stability equations are straightforward - they're the derivatives of each 6 forces and
moments with respect to alpha, beta, rotation rates, velocity, and control surface deflections.

* We don't think OpenAeroStruct has roll rates or control surface deflections, so this example could
just be alpha, beta, and freestream velocity.

* We view the test problem as having a dihedral wing and a V-tail. The objective would be to minimize
wing weight, subject to Cn-beta and Cm-alpha constraints. We can provide baseline geometry.

# Optimization Problem

Problem: eCRM-001 Wing/Tail but with addition of dihedral to wing

Objective: Maximize L/D @ 150 mph

Constraints @ 70 mph, 150 mph, 200 mph:
  -CM_α/CL_α > 0.0
  CN_β > 0.0
  CL < 1.3
  CL = W/qS

Design Variables:
  Vertical Tail Area
  Horizontal Tail Area
  Wing Chord
  α @ each speed

# Tools Needed

**VSP**

Installing vsp is covered in this [document](openmdao_with_vsp.md)

**OpenAerostruct**

OpenAerostruct can be installed by cloning the repo and installing it into an activated OpenMDAO
environment following the instructions they give.

There is a bug in the existing version of OpenAerostruct where beta (sideslip angle) is not
promoted out of an AerostructGroup. We have fixed this and will submit it back to the OAS repo at
the conclusion of the reverse-hackathon.

# Overall Approach

We felt that it made the most sense to implement this using a subproblem contained inside of a
Group. The subproblem contains an OpenMDAO problem that runs an OpenAerostruct model to compute
the aerodynamic forces and moments, and then performs a compute_totals to calculate the stability
derivatives which are the derivatives of some of the aircraft forces and moments with respect to
alpha and beta.  OAS provides efficient analytic derivatives, but VSP does not, so the VSP
component will compute its partial derivatives using finite difference.

We want to constrain the stability derivatives computed at three different operating speeds, so we
need three instances of the OAS-containing-component. These calculations don't depend on each
other, so they can be placed in a ParallelGroup, and the optimization can be run under MPI with
three processors.

In the top level optimization, we need derivatives of the stability derivatives with respect to
all of the design variables. In effect, we need the second derivartives of aerodynamic forces
and moments with respect to alpha and beta, so the component that contains the OAS subproblem
will compute its derivatives with finite-difference. (Complex step won't work here for several
reasons: VSP doesn't support complex inputs; openmdao models don't support complex inputs and
outputs for execution or computing total derivatives.

# Challenges

## 1. Modifying OpenAerostruct to use VSP as the geometry provider.

## 2. Providing a deformed mesh to OpenAerostruct in the form it expects.

## 3. OpenAerostruct scales poorly with large mesh sizes.