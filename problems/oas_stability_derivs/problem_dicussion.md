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
* -CM_α/CL_α > 0.0
*  CN_β > 0.0
*  CL < 1.3
*  CL = W/qS

Design Variables:
*  Vertical Tail Area
*  Horizontal Tail Area
*  Wing Chord
*  α @ each speed

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
total derivatives of part of your model and use them for later calculation in OpenMDAO. To be
completely flexible, these derivatives need to be an output of some kind of component so that
they can be used as inputs to other components or declared as constraint sources. However, whenver
an output is provided in OpenMDAO, the component that computes it must also be able to provide
partial derivatives of that input with respect to component inputs. So, even if we could somehow
piggyback on the full-model total derivatives capbility to compute the derivatives of a sub-model
and provide them as outputs, we would still need to compute second derivatives, and OpenMDAO
does not provide those at this time.

So this means we need to consider other approaches. One way to compute derivatives on part of
a model is to take that portion of the model and place it in a sub-problem. The portion that needs
to be compartmentalized is the set of components that are relevant between the output (the "of" in the
derivative) and the input (the "wrt" in the derivative.) As a sub-problem, we can set the new
values of the inputs, call "run_model" on it to compute the outputs, and call "compute_totals" on
it to compute the derivatives.

OpenMDAO doesn't have an "automatic" way to place problems into models. To do this, you need to create
an `ExplicitComponent` that contains the subproblem and manages the passing of data. This isn't
as difficult as it sounds, and this is one of the primary focuses of this

(Simple explanation with diagram goes here.)

For our aircraft case, we want to compute and constrain the derivatives of aerodynamic variables with
resepct to alpha and beta.

```
class ECRM(om.ExplicitComponent):
    """
    Component wrapper that contains an OpenMDAO problem with VSP and OpenAeroStruct.

    Attributes
    ----------
    _problem : <Problem>
        OpenMDAO Problem that this component runs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._problem = None

        # Added a switch to disable them when I was just doing parameter studies on the lift
        # constraint.
        self._calc_stability_derivs = True
        self._totals = {}

    def initialize(self):
        self.options.declare('wing_surface', types=dict, default={},
                             desc="Dict containing settings that define the OAS surface "
                             "for the wing.")
        self.options.declare('horiz_tail_surface', types=dict, default={},
                             desc="Dict containing settings that define the OAS surface "
                             "for the horizontal tail.")
        self.options.declare('vert_tail_surface', types=dict, default={},
                             desc="Dict containing settings that define the OAS surface "
                             "for the vertical tail.")
        self.options.declare('num_nodes', default=1,
                             desc='Number of flight points to run.')

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Design Parameters
        self.add_input('v', val=248.136 * np.ones(num_nodes), units='m/s')
        self.add_input('alpha', val=np.ones(num_nodes), units='deg')
        self.add_input('Mach_number', val=0.1 * np.ones(num_nodes))

        # VSP Geometry Parameters
        self.add_input('wing_cord', val=59.05128, units='cm')
        self.add_input('vert_tail_area', val=2295., units='cm**2')
        self.add_input('horiz_tail_area', val=6336., units='cm**2')

        # Constant Inputs
        self.add_input('beta', val=0.0, units='deg')
        self.add_input('re', val=1.0e6, units='1/m')
        self.add_input('rho', val=0.38, units='kg/m**3')
        self.add_input('CT', val=grav_constant * 17.e-6, units='1/s')
        self.add_input('R', val=50.0, units='km')
        self.add_input('W0', val=2000.0,  units='kg')
        self.add_input('speed_of_sound', val=295.4, units='m/s')
        self.add_input('load_factor', val=1.)
        self.add_input('empty_cg', np.array([262.614, 0.0, 115.861]), units='cm')

        # Outputs
        self.add_output('CL', np.zeros(num_nodes))
        self.add_output('CD', np.zeros(num_nodes))

        self.add_output('CM_alpha', np.zeros(num_nodes))
        self.add_output('CL_alpha', np.zeros(num_nodes))
        self.add_output('CN_beta', np.zeros(num_nodes))

        self.add_output('L_equals_W', np.zeros(num_nodes))
```

```
    def setup_partials(self):
        num_nodes = self.options['num_nodes']

        # This component calculates all derivatives during compute.
        self.declare_partials(of=['CL', 'CD', 'L_equals_W'],
                              wrt=['wing_cord', 'vert_tail_area', 'horiz_tail_area'],
                              rows=np.arange(num_nodes), cols=np.zeros(num_nodes))
        self.declare_partials(of=['CL', 'CD', 'L_equals_W'],
                              wrt=['alpha', 'v', 'Mach_number'],
                              rows=np.arange(num_nodes), cols=np.arange(num_nodes))

        # But we also need derivatives of the stability derivatives.
        # Those can only be computed with FD.
        self.declare_partials(of=['CM_alpha', 'CL_alpha', 'CN_beta'],
                              wrt=['alpha', 'v', 'Mach_number', 'wing_cord',
                                   'vert_tail_area', 'horiz_tail_area'],
                              method='fd', step_calc='rel')

    def compute(self, inputs, outputs):
        num_nodes = self.options['num_nodes']
        prob = self._problem

        # Build the model the first time we run.
        if prob is None:
            opt = self.options
            wing_surface = opt['wing_surface']
            horiz_tail_surface = opt['horiz_tail_surface']
            vert_tail_surface = opt['vert_tail_surface']
            surfaces = [wing_surface, horiz_tail_surface, vert_tail_surface]

            prob = om.Problem()
            model = prob.model

            # Using manual indepvarcomp because of some promote errors in OAS if I omit it.
            indep_var_comp = om.IndepVarComp()
            indep_var_comp.add_output('v', val=248.136, units='m/s')
            indep_var_comp.add_output('alpha', val=0.0, units='deg')
            indep_var_comp.add_output('beta', val=0.0, units='deg')
            indep_var_comp.add_output('Mach_number', val=0.1)
            indep_var_comp.add_output('re', val=1.0e6, units='1/m')
            indep_var_comp.add_output('rho', val=0.38, units='kg/m**3')
            indep_var_comp.add_output('CT', val=grav_constant * 17.e-6, units='1/s')
            indep_var_comp.add_output('R', val=50.0, units='km')
            indep_var_comp.add_output('W0', val=2000.0,  units='kg')
            indep_var_comp.add_output('speed_of_sound', val=295.4, units='m/s')
            indep_var_comp.add_output('load_factor', val=1.)
            indep_var_comp.add_output('empty_cg', val=np.array([262.614, 0.0, 115.861]), units='cm')

            prob.model.add_subsystem('prob_vars', indep_var_comp, promotes=['*'])

            model.add_subsystem('geo', AerostructGeometries(surfaces=surfaces),
                                promotes_inputs=['wing_cord', 'vert_tail_area', 'horiz_tail_area'])

            # Create the aero point group, which contains the actual aerodynamic
            # analyses
            aero_struct_group = AerostructPoint(surfaces=surfaces)
            prob.model.add_subsystem('aero', aero_struct_group,
                                     promotes_inputs=['v', 'alpha', 'beta', 'Mach_number', 're', 'rho',
                                                      'CT', 'R', 'W0', 'speed_of_sound', 'load_factor',
                                                      'empty_cg'])

            # Mesh Connections for AeroStruct
            for surface in surfaces:
                name = surface['name']

                prob.model.connect(f'geo.{name}.local_stiff_transformed', f'aero.coupled.{name}.local_stiff_transformed')
                prob.model.connect(f'geo.{name}.nodes', f'aero.coupled.{name}.nodes')

                # Connect aerodyamic mesh to coupled group mesh
                prob.model.connect(f'geo.{name}_mesh', f'aero.coupled.{name}.mesh')

                # Connect performance calculation variables
                for vname in ['radius', 'thickness', 'nodes']:
                    prob.model.connect(f'geo.{name}.{vname}', f'aero.{name}_perf.{vname}')

                for vname in ['cg_location', 'structural_mass', ]:
                    prob.model.connect(f'geo.{name}.{vname}', f'aero.total_perf.{name}_{vname}')

                prob.model.connect(f'geo.{name}:t_over_c', f'aero.{name}_perf.t_over_c')

                # Finite Difference is faster.
                #prob.model.approx_totals(method='fd')

            prob.setup()
            self._problem = prob

        # Set constants
        prob.set_val('beta', inputs['beta'])
        prob.set_val('re', inputs['re'])
        prob.set_val('rho', inputs['rho'])
        prob.set_val('CT', inputs['CT'])
        prob.set_val('R', inputs['R'])
        prob.set_val('W0', inputs['W0'])
        prob.set_val('speed_of_sound', inputs['speed_of_sound'])
        prob.set_val('load_factor', inputs['load_factor'])
        prob.set_val('empty_cg', inputs['empty_cg'])

        # Design inputs don't vary over cases.
        prob.set_val('wing_cord', inputs['wing_cord'])
        prob.set_val('vert_tail_area', inputs['vert_tail_area'])
        prob.set_val('horiz_tail_area', inputs['horiz_tail_area'])

        for j in range(num_nodes):

            # Set new design values.
            prob.set_val('v', inputs['v'][j])
            prob.set_val('alpha', inputs['alpha'][j])
            prob.set_val('Mach_number', inputs['Mach_number'][j])

            # Run Problem
            prob.run_model()

            # Extract Outputs
            outputs['CL'][j] = prob.get_val('aero.CL')
            outputs['CD'][j] = prob.get_val('aero.CD')
            outputs['L_equals_W'][j] = prob.get_val('aero.L_equals_W')

            if self._calc_stability_derivs:

                # Compute all component derivatives.
                # Compute Stability Derivatives and
                of = ['aero.CL', 'aero.CD', 'aero.CM', 'aero.L_equals_W']
                wrt = ['alpha', 'beta', 'v', 'Mach_number',
                       'wing_cord', 'vert_tail_area', 'horiz_tail_area']

                from time import time
                t0 = time()
                totals = prob.compute_totals(of=of, wrt=wrt)
                print('OAS stability deriv time:', time() - t0)

                # Extract Stability Derivatives
                outputs['CM_alpha'][j] = totals['aero.CM', 'alpha'][1, 0]
                outputs['CL_alpha'][j] = totals['aero.CL', 'alpha'][0, 0]
                outputs['CN_beta'][j] = totals['aero.CM', 'beta'][2, 0]

                self._totals[j] = totals

    def compute_partials(self, inputs, partials):
        num_nodes = self.options['num_nodes']
        ofs = ['CL', 'CD', 'L_equals_W']
        local_ofs = ['aero.CL', 'aero.CD', 'aero.L_equals_W']
        wrts = ['alpha', 'v', 'Mach_number',
                'wing_cord', 'vert_tail_area', 'horiz_tail_area']

        for j in range(num_nodes):
            for of, local_of in zip(ofs, local_ofs):
                for wrt in wrts:
                    partials[of, wrt][j] = self._totals[j][local_of, wrt]
```

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
and the second is span-wise. The order is important. If you find that your structural masses
are being computed as negative, then you know that you need to flip the order in one of the
dimensions.

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
