"""
OpenMDAO component wrapper for a VSP model of a modified CRM (eCRM-001) that will be used to demonstrate the
computation and use of stability derivatives in a design problem.
"""
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
            self.add_output('wing_mesh', shape=(6, 9, 3), units='inch')
            self.add_output('vert_tail_mesh', shape=(5, 5, 3), units='inch')
            self.add_output('horiz_tail_mesh', shape=(5, 5, 3), units='inch')
        else:
            # Note: at present, OAS can't handle this size.
            self.add_output('wing_mesh', shape=(23, 33, 3), units='inch')
            self.add_output('vert_tail_mesh', shape=(33, 9, 3), units='inch')
            self.add_output('horiz_tail_mesh', shape=(33, 9, 3), units='inch')

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Set values.
        vsp.SetParmVal(self.vert_tail_id, "TotalArea", "WingGeom", inputs['vert_tail_area'][0])
        vsp.SetParmVal(self.horiz_tail_id, "TotalArea", "WingGeom", inputs['horiz_tail_area'][0])
        vsp.SetParmVal(self.wing_id, "TotalChord", "WingGeom", inputs['wing_cord'][0])

        vsp.Update()
        vsp.Update()  # just in case..

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

    def vsp_to_point_cloud(self, degen_obj: degen_geom.DegenGeom)->np.ndarray:
        npts = degen_obj.surf.num_pnts
        n_xsecs = degen_obj.surf.num_secs

        points = np.empty((npts * n_xsecs, 3))
        points[:, 0] = list(itertools.chain.from_iterable(degen_obj.surf.x))
        points[:, 1] = list(itertools.chain.from_iterable(degen_obj.surf.y))
        points[:, 2] = list(itertools.chain.from_iterable(degen_obj.surf.z))

        return points


if __name__ == "__main__":

    vsp_comp = VSPeCRM(horiz_tail_name="Tail",
                       vert_tail_name="VerticalTail",
                       wing_name="Wing",
                       reduced=True)

    p = om.Problem()

    model = p.model

    p.model.add_subsystem("vsp_comp", vsp_comp)

    p.setup()

    p.run_model()

    data = {}
    for item in ['wing_mesh', 'vert_tail_mesh', 'horiz_tail_mesh']:
        data[item] = p.get_val(f"vsp_comp.{item}", units='m')

    # Save the meshes in a pickle. These will become the undeformed baseline meshes in
    # OpenAeroStruct.
    with open('baseline_meshes_reduced.pkl', 'wb') as f:
        pickle.dump(data, f)

    #om.n2(p)
    print('done')