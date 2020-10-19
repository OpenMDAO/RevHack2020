We're interested in optimizing a vehicle subject to aerodynamic stability requirements. This is a novel OpenMDAO problem because it requires constraining derivatives of components, not just values themselves.

We envision using OpenAeroStruct (since it has analytical derivatives) to drive parameters such as dihedral and taper ratio to some objective (weight, range, etc.,) subject to stability derivatives (Cnbeta, Cmalpha, etc.) being greater than some value.

The aero stability equations are straightforward - they're the derivatives of each 6 forces and moments with respect to alpha, beta, rotation rates, velocity, and control surface deflections.

We don't think OpenAeroStruct has roll rates or control surface deflections, so this example could just be alpha, beta, and freestream velocity.

We view the test problem as having a dihedral wing and a V-tail. The objective would be to minimize wing weight, subject to Cn-beta and Cm-alpha constraints. We can provide baseline geometry.
