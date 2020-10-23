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
                             desc="When True, output reduced meshes instead of full-size ones.")

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
            self.add_output('wing_mesh', shape=(12, 9, 3), units='inch')
            self.add_output('vert_tail_mesh', shape=(9, 9, 3), units='inch')
            self.add_output('horiz_tail_mesh', shape=(9, 9, 3), units='inch')
        else:
            self.add_output('wing_mesh', shape=(23, 33, 3), units='inch')
            self.add_output('vert_tail_mesh', shape=(33, 9, 3), units='inch')
            self.add_output('horiz_tail_mesh', shape=(33, 9, 3), units='inch')

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # set values
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
        #wing_cuts = self.vsp_to_cuts(degen_obj, plane='xz')
        wing_pts = self.vsp_to_point_cloud(degen_obj)

        degen_obj: degen_geom.DegenGeom = obj_dict[self.options['horiz_tail_name']]
        #horiz_tail_cuts = self.vsp_to_cuts(degen_obj, plane='xz')
        horiz_tail_pts = self.vsp_to_point_cloud(degen_obj)

        degen_obj: degen_geom.DegenGeom = obj_dict[self.options['vert_tail_name']]
        #vert_tail_cuts = self.vsp_to_cuts(degen_obj, plane='xy')
        vert_tail_pts = self.vsp_to_point_cloud(degen_obj)

        # OAS expects x stripes.
        wing_pts = wing_pts.reshape((45, 33, 3), order='F')
        horiz_tail_pts = horiz_tail_pts.reshape((33, 9, 3), order='F')
        vert_tail_pts = vert_tail_pts.reshape((33, 9, 3), order='F')

        # Meshes have symmetry pts duplicated (not mirrored.) Use half.
        wing_pts = wing_pts[:23, :, :]
        horiz_tail_pts = horiz_tail_pts[:17, :, :]
        vert_tail_pts = vert_tail_pts[:17, :, :]

        # Reduce for testing. (See John Jasa's recommendations in the docs.)
        if self.options['reduced']:
            wing_pts = wing_pts[::2, ::4, :]
            horiz_tail_pts = horiz_tail_pts[::2, :, :]
            vert_tail_pts = vert_tail_pts[::2, :, :]

        # Flip around so that FEM normals yield positive areas.
        wing_pts = wing_pts[::-1, ::-1, :]
        horiz_tail_pts = horiz_tail_pts[::-1, ::-1, :]
        vert_tail_pts = vert_tail_pts[::-1, ::-1, :]

        # outputs go here
        outputs['wing_mesh'] = wing_pts
        outputs['vert_tail_mesh'] = horiz_tail_pts
        outputs['horiz_tail_mesh'] = vert_tail_pts

    def vsp_to_cuts(self, degen_obj: degen_geom.DegenGeom, plane: str = 'xz') -> [[float]]:
        """
        Outputs sectional cuts in (eta, xle, yle, zle, twist, chord)
        :param degen_obj: degen geom object
        :param plane: plane in which to calculate the incidence angle
        :return:
        """
        # eta, xle, yle, zle, twist, chord
        s: degen_geom.DegenStick = degen_obj.sticks[0]
        ncuts = s.num_secs
        data = []
        for icut in range(ncuts):
            inc_angle = float("nan")
            if plane == 'xz':
                inc_angle = np.rad2deg(np.arcsin((s.te[icut][2] - s.le[icut][2]) / s.chord[icut]))
            elif plane == 'xy':
                inc_angle = np.rad2deg(np.arcsin((s.te[icut][1] - s.le[icut][1]) / s.chord[icut]))

            data.append(
                f'{float(icut / (ncuts - 1))},{s.le[icut][0]},{s.le[icut][1]},{s.le[icut][2]},{inc_angle},{s.chord[icut]}')

        return data

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

    with open('baseline_meshes_reduced.pkl', 'wb') as f:
        pickle.dump(data, f)

    #om.n2(p)
    print('done')