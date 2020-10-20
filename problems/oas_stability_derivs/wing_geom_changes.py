import openmdao.api as om
from openmdao.api import DirectSolver, NewtonSolver, BoundsEnforceLS
import openvsp as vsp
import degen_geom
import numpy as np
import itertools


class VSP(om.ExplicitComponent):

    def __init__(self, horiz_tail_name, vert_tail_name, wing_name):
        super().__init__()
        self.horiz_tail_name = horiz_tail_name
        self.vert_tail_name = vert_tail_name
        self.wing_name = wing_name

        self.wing_id = vsp.FindGeomsWithName(self.wing_name)[0]
        self.horiz_tail_id = vsp.FindGeomsWithName(self.horiz_tail_name)[0]
        self.vert_tail_id = vsp.FindGeomsWithName(self.vert_tail_name)[0]

    def setup(self):
        self.add_input('wing_cord', val=59.05128,)
        self.add_input('vert_tail_area', val=2295.)
        self.add_input('horiz_tail_area', val=6336.)

        self.add_output('test')
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
        #degen_obj: degen_geom.DegenGeom = list(dg.get_degen_obj_by_name(self.wing_name)[0].copies.values())[0][0]
        degen_obj: degen_geom.DegenGeom = obj_dict[self.wing_name]
        wing_cuts = self.vsp_to_cuts(degen_obj, plane='xz')
        wing_pts = self.vsp_to_point_cloud(degen_obj)

        ##degen_obj: degen_geom.DegenGeom = list(dg.get_degen_obj_by_name(self.horiz_tail_name)[0].copies.values())[0][0]
        #degen_obj: degen_geom.DegenGeom = list(dg.FindGeomsWithName(self.horiz_tail_name)[0])[0][0]
        degen_obj: degen_geom.DegenGeom = obj_dict[self.horiz_tail_name]
        horiz_tail_cuts = self.vsp_to_cuts(degen_obj, plane='xz')
        horiz_tail_pts = self.vsp_to_point_cloud(degen_obj)

        ##degen_obj: degen_geom.DegenGeom = list(dg.get_degen_obj_by_name(self.vert_tail_name)[0].copies.values())[0][0]
        #degen_obj: degen_geom.DegenGeom = list(dg.FindGeomsWithName(self.vert_tail_name)[0])[0][0]
        degen_obj: degen_geom.DegenGeom = obj_dict[self.vert_tail_name]
        vert_tail_cuts = self.vsp_to_cuts(degen_obj, plane='xy')
        vert_tail_pts = self.vsp_to_point_cloud(degen_obj)

        # outputs go here
        outputs['test'] = 1

    def vsp_to_cuts(self, degen_obj: degen_geom.DegenGeom, plane: str = 'xz') -> [[float]]:
        '''
        Outputs sectional cuts in (eta, xle, yle, zle, twist, chord)
        :param degen_obj: degen geom object
        :param plane: plane in which to calculate the incidence angle
        :return:
        '''
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
    vsp_file = 'eCRM-001.1_wing_tail.vsp3'
    vsp.ReadVSPFile(vsp_file)

    vsp_comp = VSP(horiz_tail_name="Tail", vert_tail_name="VerticalTail", wing_name="Wing")

    p = om.Problem()

    model = p.model
    newton = model.nonlinear_solver = NewtonSolver()
    newton.options['iprint'] = 2
    newton.options['maxiter'] = 100
    newton.options['solve_subsystems'] = True
    newton.options['atol'] = 1e-7
    newton.options['rtol'] = 1e-15
    newton.options['err_on_non_converge'] = True

    model.linear_solver = DirectSolver()
    ls = newton.linesearch = BoundsEnforceLS()
    ls.options['iprint'] = 2
    ls.options['print_bound_enforce'] = True
    ls.options['bound_enforcement'] = 'scalar'

    p.model.add_subsystem("vsp_comp", vsp_comp)

    p.setup()

    p.run_model()

    om.n2(p)
