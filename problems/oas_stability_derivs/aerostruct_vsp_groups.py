"""

"""
import numpy as np

import openmdao.api as om

from openaerostruct.structures.spatial_beam_setup import SpatialBeamSetup
from openaerostruct.structures.tube_group import TubeGroup
from openaerostruct.structures.wingbox_group import WingboxGroup

from vsp_eCRM import VSPeCRM


class AerostructGeometries(om.Group):
    """
    Modification of AerostructGeometry to use VSP.

    Structural analysis only happens on the wing.
    """

    def initialize(self):
        self.options.declare('surfaces', types=list)

    def setup(self):
        surfaces = self.options['surfaces']

        # Splinecomp for the thickness control points.
        for surface in surfaces:
            if 't_over_c_cp' in surface.keys():
                name = surface['name']
                n_cp = len(surface['t_over_c_cp'])
                ny = surface['mesh'].shape[1]
                x_interp = np.linspace(0., 1., int(ny-1))

                spline_comp = om.SplineComp(method='bsplines', x_interp_val=x_interp, num_cp=n_cp,
                                            interp_options={'order' : min(n_cp, 4)})

                self.add_subsystem(f'{name}_t_over_c_bsp', spline_comp,
                                   promotes_inputs=[('t_over_c_cp', f'{name}:t_over_c_cp')],
                                   promotes_outputs=[('t_over_c', f'{name}:t_over_c')])
                spline_comp.add_spline(y_cp_name='t_over_c_cp', y_interp_name='t_over_c',
                                       y_cp_val=surface['t_over_c_cp'])

        # VSP Geometry.
        self.add_subsystem('vsp', VSPeCRM(horiz_tail_name="Tail",
                                          vert_tail_name="VerticalTail",
                                          wing_name="Wing",
                                          reduced=True),
                           promotes_inputs=['wing_cord', 'vert_tail_area', 'horiz_tail_area'],
                           promotes_outputs=['wing_mesh', 'vert_tail_mesh', 'horiz_tail_mesh'])

        # Setting up the structural solve.
        for surface in surfaces:
            name = surface['name']
            sub = self.add_subsystem(name, om.Group())

            if surface['fem_model_type'] == 'tube':
                tube_promotes = []
                tube_inputs = []
                if 'thickness_cp' in surface.keys():
                    tube_promotes.append('thickness_cp')
                if 'radius_cp' not in surface.keys():
                    tube_inputs = ['mesh', 't_over_c']
                sub.add_subsystem('tube_group',
                                  TubeGroup(surface=surface, connect_geom_DVs=True),
                                  promotes_inputs=tube_inputs,
                                  promotes_outputs=['A', 'Iy', 'Iz', 'J', 'radius', 'thickness'] + tube_promotes)

            elif surface['fem_model_type'] == 'wingbox':
                wingbox_promotes = []
                if 'skin_thickness_cp' in surface.keys() and 'spar_thickness_cp' in surface.keys():
                    wingbox_promotes.append('skin_thickness_cp')
                    wingbox_promotes.append('spar_thickness_cp')
                    wingbox_promotes.append('skin_thickness')
                    wingbox_promotes.append('spar_thickness')
                elif 'skin_thickness_cp' in surface.keys() or 'spar_thickness_cp' in surface.keys():
                    raise NameError('Please have both skin and spar thickness as design variables, not one or the other.')

                sub.add_subsystem('wingbox_group',
                                  WingboxGroup(surface=surface),
                                  promotes_inputs=['mesh', 't_over_c'],
                                  promotes_outputs=['A', 'Iy', 'Iz', 'J', 'Qz', 'A_enc', 'A_int', 'htop', 'hbottom', 'hfront', 'hrear'] + wingbox_promotes)
            else:
                raise NameError('Please select a valid `fem_model_type` from either `tube` or `wingbox`.')

            if surface['fem_model_type'] == 'wingbox':
                promotes = ['A_int']
            else:
                promotes = []

            sub.add_subsystem('struct_setup',
                              SpatialBeamSetup(surface=surface),
                              promotes_inputs=['mesh', 'A', 'Iy', 'Iz', 'J'] + promotes,
                              promotes_outputs=['nodes', 'local_stiff_transformed', 'structural_mass', 'cg_location', 'element_mass'])

            self.connect(f'{name}_mesh', [f'{name}.mesh'])
            self.connect(f'{name}:t_over_c', [f'{name}.t_over_c'])
