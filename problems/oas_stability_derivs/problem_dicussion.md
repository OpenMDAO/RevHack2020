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

Problem: eCRM-001 Wing/Vertical Tail/Horizontal Tail but with addition of dihedral to wing

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

Installing vsp is covered in this [document.](openmdao_with_vsp.md)

**OpenAerostruct**

OpenAerostruct can be installed by cloning the repo and installing it into an activated OpenMDAO
environment following the instructions they give.

There is a bug in the existing version of OpenAerostruct where beta (sideslip angle) is not
promoted out of an AerostructGroup. We have fixed this and will submit it back to the OAS repo at
the conclusion of the reverse-hackathon.

# Overall Approach

In the proposed optimization problem, we want to apply constraints to stability derivatives. The
stability derivatives that we need are the derivatives of aerodynamic forces with respect to angle
of attack (alpha) and sidesplip angle (beta).  OpenMDAO can compute these derivatives, however
there is no way to use the derivatives inside of a model. Moreover, to use a derivative as a constraint,
you also need its derivatives, which are the second derivatives of the aero forces with respect
to alpha and beta. The way to accomplish this in OpenMDAO is to use a nested Problem.

# Challenges

## 1. Need an OpenMDAO component wrapper for OpenVSP.

## 2. OpenVSP's point cloud is not in the form OpenAerostruct expects.

## 3. OpenAerostruct does not directly support pluggable geometry providers.

## 4. OpenAerostruct scales poorly with large mesh sizes.

## 5. Multiple VSP instances can't run in the same model.

## 6. It is important to correctly set all inputs values for OpenAerostruct.

## 7. The optimization problem has its own challenges.

# Solution