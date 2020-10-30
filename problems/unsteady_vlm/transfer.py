""" Defines the transfer component to couple aero and struct analyses. """

import numpy

import openmdao.api as om


class TransferDisplacements(om.ExplicitComponent):
    """ Performs displacement transfer """

    def initialize(self):
        self.options.declare("nx")
        self.options.declare("n")
        self.options.declare("t")
        self.options.declare("fem_origin")

    def setup(self):
        nx = self.options["nx"]
        n = self.options["n"]
        t = self.options["t"]
        if self.options["fem_origin"] is None:
            fem_origin = 0.35
        else:
            fem_origin = self.options["fem_origin"]

        lt = t - 1  # previous time step (last dt)
        disp_lt = 'disp_%d'%lt
        def_mesh_t = 'def_mesh_%d'%t

        self.fem_origin = fem_origin

        self.add_output(def_mesh_t, val=numpy.zeros((nx, n, 3)))
        self.add_input('mesh', val=numpy.zeros((nx, n, 3)))

        if t > 0:
            self.add_input(disp_lt, val=numpy.zeros((n, 6)))

        self.num_x = nx
        self.num_y = n
        self.t = t

        self.disp = numpy.zeros((n, 6), dtype="complex")
        self.Smesh = numpy.zeros((nx, n, 3), dtype="complex")
        self.def_mesh = numpy.zeros((nx, n, 3), dtype="complex")

        self.disp_lt = disp_lt
        self.def_mesh_t = def_mesh_t

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):

        nx = self.num_x
        n = self.num_y
        t = self.t
        disp_lt = self.disp_lt
        def_mesh_t = self.def_mesh_t

        print("time step ", t)

        if t > 0:
            self.disp = inputs[disp_lt]

        w = self.fem_origin
        ref_curve = (1-w) * inputs['mesh'][0, :, :] + w * inputs['mesh'][-1, :, :]

        for ind in range(nx):
            self.Smesh[ind, :, :] = inputs['mesh'][ind, :, :] - ref_curve

        cos, sin = numpy.cos, numpy.sin
        for ind in range(n):
            dx, dy, dz, rx, ry, rz = self.disp[ind, :]

            # 1 eye from the axis rotation matrices
            # -3 eye from subtracting Smesh three times
            T = -2 * numpy.eye(3, dtype="complex")
            T[ 1:,  1:] += [[cos(rx), sin(rx)], [-sin(rx), cos(rx)]]
            T[::2, ::2] += [[cos(ry),-sin(ry)], [ sin(ry), cos(ry)]]
            T[ :2,  :2] += [[cos(rz), sin(rz)], [-sin(rz), cos(rz)]]

            self.def_mesh[:, ind, :] += self.Smesh[:, ind, :].dot(T)
            self.def_mesh[:, ind, 0] += dx
            self.def_mesh[:, ind, 1] += dy
            self.def_mesh[:, ind, 2] += dz

            outputs[def_mesh_t] = self.def_mesh + inputs['mesh']


class TransferLoads(om.ExplicitComponent):
    """ Performs load transfer """

    def initialize(self):
        self.options.declare("nx")
        self.options.declare("n")
        self.options.declare("t")
        self.options.declare("fem_origin")

    def setup(self):
        nx = self.options["nx"]
        n = self.options["n"]
        t = self.options["t"]
        if self.options["fem_origin"] is None:
            fem_origin = 0.35
        else:
            fem_origin = self.options["fem_origin"]

        # t is the actual time step (actual dt)
        def_mesh_t = 'def_mesh_%d'%t
        sec_forces_t = 'sec_forces_%d'%t
        loads_t = 'loads_%d'%t

        self.add_input(def_mesh_t, val=numpy.zeros((nx, n, 3)))
        self.add_input(sec_forces_t, val=numpy.zeros((n-1, 3), dtype="complex"))
        self.add_output(loads_t, val=numpy.zeros((n, 6), dtype="complex"))

        self.n = n
        self.t = t
        self.fem_origin = fem_origin

        self.moment = numpy.zeros((n-1, 3), dtype="complex")
        self.loads = numpy.zeros((n, 6), dtype="complex")

        self.def_mesh_t = def_mesh_t
        self.sec_forces_t = sec_forces_t
        self.loads_t = loads_t

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):

        def_mesh_t = self.def_mesh_t
        sec_forces_t = self.sec_forces_t
        loads_t = self.loads_t

        mesh = inputs[def_mesh_t]
        sec_forces = inputs[sec_forces_t]

        w = 0.25
        a_pts = 0.5 * (1-w) * mesh[:-1, :-1, :] + \
                0.5 *   w   * mesh[1:, :-1, :] + \
                0.5 * (1-w) * mesh[:-1,  1:, :] + \
                0.5 *   w   * mesh[1:,  1:, :]

        w = self.fem_origin
        s_pts = 0.5 * (1-w) * mesh[:-1, :-1, :] + \
                0.5 *   w   * mesh[1:, :-1, :] + \
                0.5 * (1-w) * mesh[:-1,  1:, :] + \
                0.5 *   w   * mesh[1:,  1:, :]

        for ind in range(numpy.int(self.n-1)):
            r = a_pts[0, ind, :] - s_pts[0, ind, :]
            F = sec_forces[ind, :]
            self.moment[ind, :] = numpy.cross(r, F)

        self.loads[:-1, :3] += 0.5 * sec_forces[:, :]
        self.loads[ 1:, :3] += 0.5 * sec_forces[:, :]
        self.loads[:-1, 3:] += 0.5 * self.moment[:, :]
        self.loads[ 1:, 3:] += 0.5 * self.moment[:, :]

        outputs[loads_t] = self.loads
