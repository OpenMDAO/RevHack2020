""" Manipulate geometry mesh based on high-level design parameters. """

import numpy
from numpy import cos, sin, tan

import openmdao.api as om

from bspline import get_bspline_mtx
from crm_data import crm_base_mesh


def rotate(mesh, thetas):
    """ Computes rotation matricies given mesh and rotation angles in degress. """
    te = mesh[-1]
    le = mesh[ 0]
    quarter_chord = 0.25*te + 0.75*le

    ny = mesh.shape[1]
    nx = mesh.shape[0]

    rad_thetas = thetas * numpy.pi / 180.

    mats = numpy.zeros((ny, 3, 3), dtype="complex")
    mats[:, 0, 0] = cos(rad_thetas)
    mats[:, 0, 2] = sin(rad_thetas)
    mats[:, 1, 1] = 1
    mats[:, 2, 0] = -sin(rad_thetas)
    mats[:, 2, 2] = cos(rad_thetas)

    for ix in range(nx):
        row = mesh[ix]
        row[:] = numpy.einsum("ikj, ij -> ik", mats, row - quarter_chord)
        row += quarter_chord
    return mesh


def sweep(mesh, angle):
    """ Shearing sweep angle. Positive sweeps back. """

    num_x, num_y, _ = mesh.shape
    ny2 = int((num_y-1)/2)

    le = mesh[0]
    te = mesh[-1]

    y0 = le[ny2, 1]

    tan_theta = tan(numpy.radians(angle))
    dx_right = (le[ny2:, 1] - y0) * tan_theta
    dx_left = -(le[:ny2, 1] - y0) * tan_theta
    dx = numpy.hstack((dx_left, dx_right))

    for i in range(num_x):
        mesh[i, :, 0] += dx

    return mesh


def dihedral(mesh, angle):
    """ Dihedral angle. Positive bends up. """

    num_x, num_y, _ = mesh.shape
    ny2 = int((num_y+1)/2)

    le = mesh[0]
    te = mesh[-1]

    y0 = le[ny2, 1]

    tan_theta = tan(numpy.radians(angle))
    dx_right = (le[ny2:, 1] - y0) * tan_theta
    dx_left = -(le[:ny2, 1] - y0) * tan_theta
    dx = numpy.hstack((dx_left, dx_right))

    for i in range(num_x):
        mesh[i, :, 2] += dx

    return mesh


def stretch(mesh, length):
    """ Strech mesh in span-wise direction to reach specified length. """

    le = mesh[0]
    te = mesh[-1]

    num_x, num_y, _ = mesh.shape

    span = le[-1, 1] - le[0, 1]
    dy = (length - span) / (num_y - 1) * numpy.arange(1, num_y)

    for i in range(num_x):
        mesh[i, 1:, 1] += dy

    return mesh


def taper(mesh, taper_ratio):
    """ Change the spanwise chord to produce a tapered wing. """

    le = mesh[0]
    te = mesh[-1]
    num_x, num_y, _ = mesh.shape
    ny2 = int((num_y+1)/2)

    center_chord = .5 * te + .5 * le
    span = le[-1, 1] - le[0, 1]
    taper = numpy.linspace(1, taper_ratio, ny2)[::-1]

    jac = get_bspline_mtx(ny2, ny2, mesh, order=2)
    taper = jac.dot(taper)

    dx = numpy.hstack((taper, taper[::-1][1:]))

    for i in range(num_x):
        for ind in range(3):
            mesh[i, :, ind] = (mesh[i, :, ind] - center_chord[:, ind]) * \
                dx + center_chord[:, ind]

    return mesh


def mirror(mesh, right_side=True):
    """ Takes a half geometry and mirrors it across the symmetry plane.
    If right_side==True, it mirrors from right to left, assuming that
    the first point is on the symmetry plane. Else it mirrors from left
    to right, assuming the last point is on the symmetry plane. """

    num_x, num_y, _ = mesh.shape
    new_mesh = numpy.empty((num_x, 2 * num_y - 1, 3))
    mirror_y = numpy.ones(mesh.shape)
    mirror_y[:, :, 1] *= -1.0

    if right_side:
        new_mesh[:, :num_y, :] = mesh[:, ::-1, :] * mirror_y
        new_mesh[:, num_y:, :] = mesh[:,   1:, :]
    else:
        new_mesh[:, :num_y, :] = mesh[:, ::-1, :]
        new_mesh[:, num_y:, :] = mesh[:,   1:, :] * mirror_y[:, 1:, :]

    return new_mesh


def gen_crm_mesh(n_points_inboard=2, n_points_outboard=2, num_x=2, mesh=crm_base_mesh):
    """ Builds the right hand side of the CRM wing with specified number
    of inboard and outboard panels. """

    # LE pre-yehudi
    s1 = (mesh[0, 1, 0] - mesh[0, 0, 0]) / (mesh[0, 1, 1] - mesh[0, 0, 1])
    o1 = mesh[0, 0, 0]

    # TE pre-yehudi
    s2 = (mesh[1, 1, 0] - mesh[1, 0, 0]) / (mesh[1, 1, 1] - mesh[1, 0, 1])
    o2 = mesh[1, 0, 0]

    # LE post-yehudi
    s3 = (mesh[0, 2, 0] - mesh[0, 1, 0]) / (mesh[0, 2, 1] - mesh[0, 1, 1])
    o3 = mesh[0, 2, 0] - s3 * mesh[0, 2, 1]

    # TE post-yehudi
    s4 = (mesh[1, 2, 0] - mesh[1, 1, 0]) / (mesh[1, 2, 1] - mesh[1, 1, 1])
    o4 = mesh[1, 2, 0] - s4 * mesh[1, 2, 1]

    n_points_total = n_points_inboard + n_points_outboard - 1
    half_mesh = numpy.zeros((2, n_points_total, 3))

    # generate inboard points
    dy = (mesh[0, 1, 1] - mesh[0, 0, 1]) / (n_points_inboard - 1)
    for i in range(n_points_inboard):
        y = half_mesh[0, i, 1] = i * dy
        half_mesh[0, i, 0] = s1 * y + o1 # le point
        half_mesh[1, i, 1] = y
        half_mesh[1, i, 0] = s2 * y + o2 # te point

    yehudi_break = mesh[0, 1, 1]
    # generate outboard points
    dy = (mesh[0, 2, 1] - mesh[0, 1, 1]) / (n_points_outboard - 1)
    for j in range(n_points_outboard):
        i = j + n_points_inboard - 1
        y = half_mesh[0, i, 1] = j * dy + yehudi_break
        half_mesh[0, i, 0] = s3 * y + o3 # le point
        half_mesh[1, i, 1] = y
        half_mesh[1, i, 0] = s4 * y + o4 # te point

    full_mesh = mirror(half_mesh)
    full_mesh = add_chordwise_panels(full_mesh, num_x)
    return full_mesh


def add_chordwise_panels(mesh, num_x):
    """ Divides the wing into multiple chordwise panels. """
    le = mesh[ 0, :, :]
    te = mesh[-1, :, :]

    new_mesh = numpy.zeros((num_x, mesh.shape[1], 3))
    new_mesh[ 0, :, :] = le
    new_mesh[-1, :, :] = te

    for i in range(1, num_x-1):
        w = float(i) / (num_x - 1)
        new_mesh[i, :, :] = (1 - w) * le + w * te

    return new_mesh


def gen_mesh(num_x, num_y, span, chord, cosine_spacing=0.):
    """ Builds the right hand side of the rectangular wing. """
    mesh = numpy.zeros((num_x, num_y, 3))
    ny2 = (num_y + 1) // 2
    beta = numpy.linspace(0, numpy.pi/2, ny2)

    # mixed spacing with w as a weighting factor
    cosine = .5 * numpy.cos(beta) #  cosine spacing
    uniform = numpy.linspace(0, .5, ny2)[::-1] #  uniform spacing
    half_wing = cosine * cosine_spacing + (1 - cosine_spacing) * uniform
    full_wing = numpy.hstack((-half_wing[:-1], half_wing[::-1])) * span

    for ind_x in range(num_x):
        for ind_y in range(num_y):
            mesh[ind_x, ind_y, :] = [ind_x / (num_x-1) * chord - chord/2., full_wing[ind_y], 0]

    return mesh


class GeometryMesh(om.ExplicitComponent):
    """ Changes a given mesh with span, swee3p, and twist des-vars.
    Takes in a half mesh with symmetry plane about the middle and
    outputs a full symmetric mesh. """

    def initialize(self):
        self.options.declare("mesh")
        self.options.declare("num_twist")

    def setup(self):
        self.mesh = mesh = self.options["mesh"]
        self.num_twist = num_twist = self.options["num_twist"]

        self.new_mesh = numpy.empty(mesh.shape, dtype=complex)
        self.new_mesh[:] = mesh
        self.n = self.mesh.shape[1]
        self.half = half = int((self.n-1)/2)
        self.add_input('span', val=58.7630524)
        self.add_input('sweep', val=0.)
        self.add_input('dihedral', val=0.)
        self.add_input('twist', val=numpy.zeros(num_twist))
        self.add_input('taper', val=1.)
        self.add_output('mesh', val=self.mesh[:, :half+1, :])

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        jac = get_bspline_mtx(self.num_twist, self.n, self.mesh)
        h_cp = inputs['twist']
        h = jac.dot(h_cp)

        self.new_mesh[:] = self.mesh
        stretch(self.new_mesh, inputs['span'])
        sweep(self.new_mesh, inputs['sweep'])
        rotate(self.new_mesh, h)
        dihedral(self.new_mesh, inputs['dihedral'])
        taper(self.new_mesh, inputs['taper'])
        outputs['mesh'] = self.new_mesh[:, :self.half+1, :]
