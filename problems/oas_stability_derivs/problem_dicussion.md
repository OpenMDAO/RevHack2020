# Original Problem Statement

* We're interested in optimizing a vehicle subject to aerodynamic stability requirements. This is a
novel OpenMDAO problem because it requires constraining derivatives of components, not just
values themselves.

* We envision using OpenAerostruct (since it has analytical derivatives) to drive parameters such as
dihedral and taper ratio to some objective (weight, range, etc.,) subject to stability derivatives
(Cnbeta, Cmalpha, etc.) being greater than some value.

* The aero stability equations are straightforward - they're the derivatives of each 6 forces and
moments with respect to alpha, beta, rotation rates, velocity, and control surface deflections.

* We don't think OpenAerostruct has roll rates or control surface deflections, so this example could
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

## 1. How can OpenMDAO provide derivatives to use as outputs.

This problem was proposed to the workshop in order to answer the question of how to compute
total derivatives of part of your model and use them for later calculation in OpenMDAO.

## 2. Need an OpenMDAO component wrapper for OpenVSP.

We were provided with an OpenVSP model of the eCRM-001 aerodynamic surfaces including the wing, the
vertical tail, and the horizontal tail. We were also provided with a sample OpenMDAO component
wrapper that did not fully work because it expected an API that had been modified on a forked
version of OpenVSP. However, we were able to convert the API calls to the correct ones to enable
OpenVSP to be used from OpenMDAO.

The OpenVSP geometry is parameterized with three inputs: chord length for the wing, and area for
each tail. The OpenMDAO component exposes these as inputs to allow the optimizer to vary them. The
component outputs a mesh for each aerodynamic surface. These meshes will be passed directly to
the structural and aerodynamic analyses inside of OpenAerostruct.

Here are the `initialize` and `setup` methods of the OpenVSP component.
```
import itertools
import pickle

import numpy as np

import openmdao.api as om

import openvsp as vsp
import degen_geom

class VSPeCRM(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('horiz_tail_name', default='Tail',
                             desc="Name of the horizontal tail in the vsp model.")
        self.options.declare('vert_tail_name', default='VerticalTail',
                             desc="Name of the vertical tail in the vsp model.")
        self.options.declare('wing_name', default='Wing',
                             desc="Name of the wing in the vsp model.")
        self.options.declare('reduced', default=False,
                             desc="When True, output reduced meshes instead of full-size ones. "
                             "Running with a smaller mesh is of value when debugging.")

    def setup(self):
        options = self.options
        horiz_tail_name = options['horiz_tail_name']
        vert_tail_name = options['vert_tail_name']
        wing_name = options['wing_name']
        reduced = options['reduced']

        # Read the geometry.
        vsp_file = 'eCRM-001.1_wing_tail.vsp3'
        vsp.ReadVSPFile(vsp_file)

        self.wing_id = vsp.FindGeomsWithName(wing_name)[0]
        self.horiz_tail_id = vsp.FindGeomsWithName(horiz_tail_name)[0]
        self.vert_tail_id = vsp.FindGeomsWithName(vert_tail_name)[0]

        self.add_input('wing_cord', val=59.05128,)
        self.add_input('vert_tail_area', val=2295.)
        self.add_input('horiz_tail_area', val=6336.)

        # Shapes are pre-determined.
        if reduced:
            self.add_output('wing_mesh', shape=(6, 9, 3), units='cm')
            self.add_output('vert_tail_mesh', shape=(5, 5, 3), units='cm')
            self.add_output('horiz_tail_mesh', shape=(5, 5, 3), units='cm')
        else:
            # Note: at present, OAS can't handle this size.
            self.add_output('wing_mesh', shape=(23, 33, 3), units='cm')
            self.add_output('vert_tail_mesh', shape=(33, 9, 3), units='cm')
            self.add_output('horiz_tail_mesh', shape=(33, 9, 3), units='cm')

        self.declare_partials(of='*', wrt='*', method='fd')
```
The geometry file is read in during setup. Notice that this component will use finite difference
to calculate its partial derivatives. OpenVSP does not provide analytic derivatives and does not
support complex inputs, so fd is the only option.  VSP runs pretty quickly, and the component only
has three inputs, so the derivative computation won't be a bottlneck. Also, the rest of
OpenAerostruct will continue to use analytic derivatives.

Next is the `compute` method:
```
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Set values.
        vsp.SetParmVal(self.vert_tail_id, "TotalArea", "WingGeom", inputs['vert_tail_area'][0])
        vsp.SetParmVal(self.horiz_tail_id, "TotalArea", "WingGeom", inputs['horiz_tail_area'][0])
        vsp.SetParmVal(self.wing_id, "TotalChord", "WingGeom", inputs['wing_cord'][0])

        vsp.Update()
        #vsp.Update()  # just in case..

        # run degen geom to get measurements
        dg:degen_geom.DegenGeomMgr = vsp.run_degen_geom(set_index=vsp.SET_ALL)
        obj_dict = {p.name:p for p in dg.get_all_objs()}

        # pull measurements out of degen_geom api
        degen_obj: degen_geom.DegenGeom = obj_dict[self.options['wing_name']]
        wing_cloud = self.vsp_to_point_cloud(degen_obj)

        degen_obj: degen_geom.DegenGeom = obj_dict[self.options['horiz_tail_name']]
        horiz_cloud = self.vsp_to_point_cloud(degen_obj)

        degen_obj: degen_geom.DegenGeom = obj_dict[self.options['vert_tail_name']]
        vert_cloud = self.vsp_to_point_cloud(degen_obj)
```
Here, we set the geometry parameter values into VSP, and then call `run_degen_geom` to compute
the deformed geometry. Extracting a point cloud for each surface is done by:
```
    def vsp_to_point_cloud(self, degen_obj: degen_geom.DegenGeom)->np.ndarray:
        npts = degen_obj.surf.num_pnts
        n_xsecs = degen_obj.surf.num_secs

        points = np.empty((npts * n_xsecs, 3))
        points[:, 0] = list(itertools.chain.from_iterable(degen_obj.surf.x))
        points[:, 1] = list(itertools.chain.from_iterable(degen_obj.surf.y))
        points[:, 2] = list(itertools.chain.from_iterable(degen_obj.surf.z))

        return points
```
The point clouds now need to be converted into a mesh array that can be used by OpenAerostruct.

## 3. OpenVSP's point cloud is not in the form OpenAerostruct expects.

From OpenVSP we have extracted a cloud of points for each aerodynamic surface. These aren't just
random fields of points though. The points of each cross section from the tip to the symmetry
plane are stored sequentially, so some reshaping and reordering is needed to prepare them for
OpenAerostruct, which expects a three-dimensional array where the first dimension is chord-wise
and the second is span-wise.

The OpenVSP meshes are also too large for OpenAerostruct to effectively handle. The reason for
this will be discussed further below, but it is also important to recognize that the level of
fidelity for the vortex lattice and simple structural solver are mor conducive to smaller mesh
sizes. We reduce the number of mesh points by taking every second or fourth point where symmetry
allows. We had to skip a point near the back of the wing because the mesh couldn't be subdivided
by four in that direction.

Finally, the VSP surface is actually 3-dimensional, and the point cloud includes the points on
both the upper and lower surfaces (or in the case of the veritcal tail, the left and right
surfaces.)  Both the structural and aerodynamics analyses in OpenAerostruct use two-dimensional
meshes augmented with a separate thickness variable controlled by a bsplines component. We
take the upper and lower (or left and right) surfaces and average them to make the 2D mesh.

The final code to convert the point clouds into deformed meshes is as follows:
```
        # VSP outputs wing outer mold lines at points along the span.
        # Reshape to (chord, span, dimension)
        wing_cloud = wing_cloud.reshape((45, 33, 3), order='F')
        horiz_cloud = horiz_cloud.reshape((33, 9, 3), order='F')
        vert_cloud = vert_cloud.reshape((33, 9, 3), order='F')

        # Meshes have upper and lower surfaces, so we average the z (or y for vertical).
        wing_pts = wing_cloud[:23, :, :]
        wing_pts[1:-1, :, 2] = 0.5 * (wing_cloud[-2:-23:-1, :, 2] + wing_pts[1:-1, :, 2])
        horiz_tail_pts = horiz_cloud[:17, :, :]
        horiz_tail_pts[1:-1, :, 2] = 0.5 * (horiz_cloud[-2:-17:-1, :, 2] + horiz_tail_pts[1:-1, :, 2])
        vert_tail_pts = vert_cloud[:17, :, :]
        vert_tail_pts[1:-1, :, 1] = 0.5 * (vert_cloud[-2:-17:-1, :, 1] + vert_tail_pts[1:-1, :, 1])

        # Reduce the mesh size for testing. (See John Jasa's recommendations in the docs.)
        if self.options['reduced']:
            wing_pts = wing_pts[:, ::4, :]
            wing_pts = wing_pts[[0, 4, 8, 12, 16, 22], ...]
            horiz_tail_pts = horiz_tail_pts[::4, ::2, :]
            vert_tail_pts = vert_tail_pts[::4, ::2, :]

        # Flip around so that FEM normals yield positive areas.
        wing_pts = wing_pts[::-1, ::-1, :]
        horiz_tail_pts = horiz_tail_pts[::-1, ::-1, :]
        vert_tail_pts = vert_tail_pts[:, ::-1, :]

        # outputs go here
        outputs['wing_mesh'] = wing_pts
        outputs['vert_tail_mesh'] = vert_tail_pts
        outputs['horiz_tail_mesh'] = horiz_tail_pts
```
Both the vertical and horizontal tail are symmetric, and vsp only gives us the points on one side
of the symmetry plane. OpenAerostruct also supports this provided we set the symmetry flag to
True in the surface dictionary. The vertical tail lies along the symmetry line, and VSP gives us
the entire surface.

## 4. OpenAerostruct does not directly support pluggable geometry providers.

## 5. OpenAerostruct scales poorly with large mesh sizes.

## 6. Multiple VSP instances can't run in the same model.


# Solution

## 7. It is important to correctly set all inputs values for OpenAerostruct.

## 8. The optimization problem has its own challenges.
