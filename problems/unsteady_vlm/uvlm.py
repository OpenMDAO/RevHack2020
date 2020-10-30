""" Defines the aerodynamic analysis component using Unsteady Vortex Lattice Method (UVLM) """

import matplotlib.pyplot as plt
import numpy

import openmdao.api as om
from scipy.linalg import lu_factor, lu_solve

try:
    import lib
    fortran_flag = True
except:
    fortran_flag = False

def view_mat(mat):
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = numpy.sum(mat, axis=2)
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

def norm(vec):
    """ Define the norm of the vector """
    return numpy.sqrt(numpy.sum(vec**2))

def calc_vorticity(A, B, P):
    """ Define the vorticity induced by the segment AB to P """
    rAP = P - A
    rBP = P - B
    rAP_len = norm(rAP)
    rBP_len = norm(rBP)
    cross = numpy.cross(rAP, rBP)
    may = numpy.sum(cross**2)

    # Cutoff to avoid singularity on b_pts, I think when collocation is on
    # a vortex
    r_cut = 1e-10
    cond = any([rAP_len < r_cut, rBP_len < r_cut, may < r_cut])
    if cond:
        return numpy.array([0., 0., 0.])

    return (rAP_len + rBP_len) * cross / \
           (rAP_len * rBP_len * (rAP_len * rBP_len + rAP.dot(rBP)))

def assemble_AIC_matrices(rings_mesh, points, a_mtx, b_mtx=numpy.array([]), also_b=False, wake=False):
    """ Compute the influence that the ring vortex elements of the mesh have on a list of point c_pts
    The resultant matrice (a_mtx) is a matrix of influence coefficients.
    The matrice b_mtx can also be computed for v_wing induced velocity computation.
    b_mtx takes in count only chordwise vortex segments and the last mesh row that corresponds
    to the latest unsteady wake element """

    if fortran_flag:
        a_mtx, b_mtx = lib.ringsmtx(rings_mesh, points, also_b, True)

    else:
        m_nx = rings_mesh.shape[0]
        m_ny = rings_mesh.shape[1]
        c_nx = points.shape[0]
        c_ny = points.shape[1]

        # Chordwise loop through ring elements
        for el_i in range(m_nx - 1):
            el_loc_i = el_i * (m_ny - 1)

            # Spanwise loop through ring elements
            for el_j in range(m_ny - 1):
                el_loc = el_j + el_loc_i

                # Chordwise loop through points
                for cp_i in range(c_nx):
                    cp_loc_i = cp_i * (c_ny)

                    # Spanwise loop through points
                    for cp_j in range(c_ny):
                        cp_loc = cp_j + cp_loc_i

                        P = points[cp_i, cp_j]                              # local point

                        A = rings_mesh[el_i, el_j + 0, :]           # ring element corners
                        B = rings_mesh[el_i, el_j + 1, :]
                        C = rings_mesh[el_i + 1, el_j + 1, :]
                        D = rings_mesh[el_i + 1, el_j + 0, :]

                        AB_vort = calc_vorticity(A, B, P)           # vortex segments
                        BC_vort = calc_vorticity(B, C, P)
                        CD_vort = calc_vorticity(C, D, P)
                        DA_vort = calc_vorticity(D, A, P)

                        # Influence of the other half wing (symmetry)
                        sym = numpy.array([1., -1., 1.])
                        P_sym = P * sym

                        AB_vort_sym = calc_vorticity(A, B, P_sym) * sym
                        BC_vort_sym = calc_vorticity(B, C, P_sym) * sym
                        CD_vort_sym = calc_vorticity(C, D, P_sym) * sym
                        DA_vort_sym = calc_vorticity(D, A, P_sym) * sym

                        # Computation of the influence coefficient that links the point with the element
                        a_mtx[cp_loc, el_loc, :] = AB_vort + BC_vort + CD_vort + DA_vort \
                                   + AB_vort_sym + BC_vort_sym + CD_vort_sym + DA_vort_sym

                        if also_b:
                            b_mtx[cp_loc, el_loc, :] = BC_vort + DA_vort + BC_vort_sym + DA_vort_sym
                            # Add influence of latest unsteady wake element
                            # (only in b_mtx, for induced drag computation)
                            if (el_i == m_nx - 2):
                                b_mtx[cp_loc, el_loc, :] += CD_vort + CD_vort_sym

        a_mtx /= 4 * numpy.pi
        b_mtx /= 4 * numpy.pi

    return a_mtx, b_mtx


class Geometry(om.ExplicitComponent):
    """ Compute various geometric properties for VLM analysis
    b_pts = vortex rings corners defined at quarters of panel 'chords'
    c_pts = collocation points defined at three-quarters of panel 'chords'
    starting_vortex = last row of b_pts matrix, representing the wake at t=0
    lengths = length of aerodynamic panels
    widths = widths of aerodynamic panels
    normals = normal vectors of aerodynamic panels
    v_local = local velocity of c_pts in the inertial frame
    S_ref = surface area of the wing """

    def initialize(self):
        self.options.declare("nx")
        self.options.declare("n")
        self.options.declare("t")

    def setup(self):
        nx = self.options["nx"]
        n = self.options["n"]
        t = self.options["t"]

        self.add_input('v', val=10.)
        self.add_input('alpha', val=3.)
        self.add_input('dt', val=0.1)
        self.add_input('def_mesh_' + str(t), val=numpy.zeros((nx, n, 3)))
        self.add_output('b_pts_' + str(t), val=numpy.zeros((nx, n, 3)))
        self.add_output('c_pts_' + str(t), val=numpy.zeros((nx-1, n-1, 3)))
        self.add_output('starting_vortex_' + str(t), val=numpy.zeros((1, n, 3)))
        self.add_output('lengths_' + str(t), val=numpy.zeros((nx-1, n-1)))
        self.add_output('widths_' + str(t), val=numpy.zeros((nx-1, n-1)))
        self.add_output('normals_' + str(t), val=numpy.zeros((nx-1, n-1, 3)))
        self.add_output('v_local_' + str(t), val=numpy.zeros((nx-1, n-1, 3)))
        if t == 0:
            self.add_output('S_ref', val=0.)

        self.num_x = nx
        self.num_y = n
        self.t = t

        self.unrotated_b = numpy.zeros((nx, n, 3))
        self.unrotated_c = numpy.zeros((nx-1, n-1, 3))
        self.all_lengths = numpy.zeros((nx-1, n), dtype="complex")
        self.surf = val=numpy.zeros((nx-1, n-1))

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd', form='central')

    def get_lengths(self, A, B, axis):
        return numpy.sqrt(numpy.sum((B - A)**2, axis=axis))

    def compute(self, inputs, outputs):

        nx = self.num_x
        n = self.num_y
        t = self.t

        mesh = outputs['def_mesh_' + str(t)]

        # distance of the last row of points chordwise from the trailing edge
        # last row of b_pts, defined with dist. This is also the starting vortex line
        dist_x = 0.3 * outputs['v'] * outputs['dt']

        # Set all unrotated_b points based on the mesh (as is currently done in OAS),
        # except the last one which will be based on the velocity and timestep
        self.unrotated_b[:-1, :, :] = mesh[:-1, :, :] * .75 + mesh[1:, :, :] * .25

        # Set last row of unrotated_b as the mesh itself then offset it based on the dist_x
        self.unrotated_b[-1, :, :] = mesh[-1, :, :]
        self.unrotated_b[-1, :, 0] += dist_x

        # Rotation of b_pts and c_pts due to the angle of attack alpha
        alpha_conv = outputs['alpha'] * numpy.pi / 180.
        cosa = numpy.cos(-alpha_conv)
        sina = numpy.sin(-alpha_conv)
        rot_x = numpy.array([cosa, 0, -sina])
        rot_z = numpy.array([sina, 0,  cosa])

        b_unrot = self.unrotated_b

        # Actually rotate the b_pts
        for i in range(nx):
            for j in range(n):
                outputs['b_pts_' + str(t)][i, j, 0] = b_unrot[i, j, :].dot(rot_x)
                outputs['b_pts_' + str(t)][i, j, 1] = b_unrot[i, j, 1]
                outputs['b_pts_' + str(t)][i, j, 2] = b_unrot[i, j, :].dot(rot_z)

        # Set the initial c_pts
        self.unrotated_c = 0.5 * 0.25 * mesh[:-1,:-1, :] + \
                           0.5 * 0.75 * mesh[ 1:,:-1, :] + \
                           0.5 * 0.25 * mesh[:-1, 1:, :] + \
                           0.5 * 0.75 * mesh[ 1:, 1:, :]

        c_unrot = self.unrotated_c

        # Actually rotate the c_pts
        for i in range(nx-1):
            for j in range(n-1):
                outputs['c_pts_' + str(t)][i, j, 0] = c_unrot[i, j, :].dot(rot_x)
                outputs['c_pts_' + str(t)][i, j, 1] = c_unrot[i, j, 1]
                outputs['c_pts_' + str(t)][i, j, 2] = c_unrot[i, j, :].dot(rot_z)

        # Set the starting vortex based on the last row of the b_pts; used in wake calculations
        outputs['starting_vortex_' + str(t)] = outputs['b_pts_' + str(t)][-1, :, :]

        # Get lengths of panels, used later to compute forces
        self.all_lengths[:-1, :] = self.get_lengths(b_unrot[1:-1, :, :], b_unrot[:-2, :, :], 2)

        # The final length should be the chordwise length, same as other panels
        self.all_lengths[-1, :] = self.get_lengths(mesh[-1, :, :] - b_unrot[-2, :, :], \
                                     mesh[0, :, :] - b_unrot[0, :, :], 1)

        # Save the averaged lengths to get lengths per panel
        outputs['lengths_' + str(t)] = (self.all_lengths[:, 1:] + self.all_lengths[:, :-1])/2

        # Save the widths of each panel
        outputs['widths_' + str(t)] = self.get_lengths(b_unrot[:-1, 1:, :], b_unrot[:-1, :-1, :], 2)

        # Normals to the panels
        normals = numpy.cross(
        outputs['b_pts_' + str(t)][:-1,  1:, :] - outputs['b_pts_' + str(t)][ 1:, :-1, :],
        outputs['b_pts_' + str(t)][:-1, :-1, :] - outputs['b_pts_' + str(t)][ 1:,  1:, :], axis=2)

        # Normalize the normals
        norms = numpy.sqrt(numpy.sum(normals**2, axis=2))
        for ind in range(3):
            normals[:, :, ind] /= norms
        outputs['normals_' + str(t)] = normals

        # Panel velocity in the inertial frame
        outputs['v_local_' + str(t)][:, :, 0] = outputs['v']

        # Surface area of the half wing if this is the first timestep
        if t == 0:
            surf = outputs['lengths_' + str(t)] * outputs['widths_' + str(t)]
            outputs['S_ref'] = numpy.sum(surf)

class Circulations(om.ExplicitComponent):
    """ Define wing and wake panels circulations
    'circ_' + str(t) = wing panel circulations, at the time step t
    'circ_wake_' + str(t) = wake panel circulations, at the time step t
    'v_wake_on_wing_' + str(t) = velocity of c_pts (wing panels) induced by wake panels, at the time step t """

    def initialize(self):
        self.options.declare("nx")
        self.options.declare("n")
        self.options.declare("nw")
        self.options.declare("t")

    def setup(self):
        nx = self.options["nx"]
        n = self.options["n"]
        nw = self.options["nw"]
        t = self.options["t"]

        lt = t - 1

        if t < nw:
            iw = t
        else:
            iw = nw

        self.add_input('dt', val=0.1)
        self.add_input('c_pts_' + str(t), val=numpy.zeros((nx-1, n-1, 3)))
        self.add_input('b_pts_' + str(t), val=numpy.zeros((nx, n, 3)))
        self.add_input('normals_' + str(t), val=numpy.zeros((nx-1, n-1, 3), dtype="complex"))
        self.add_input('v_local_' + str(t), val=numpy.zeros((nx-1, n-1, 3), dtype="complex"))
        size = (n-1) * (nx-1)
        self.add_output('circ_' + str(t), val=numpy.zeros((size), dtype="complex"))

        # wake circulations
        # in the first step (dt=0) there is no wake and also there aren't previous results
        # that have to be passed as input
        # in dt=1 there is the first evaluation of wake circulations
        size_wake = (n-1) * t
        size_a1 = (n-1) * iw
        if t > 0:
            self.add_input('wake_mesh_' + str(t), val=numpy.zeros((t+1, n, 3), dtype="complex"))
            self.add_input('circ_' + str(lt), val=numpy.zeros((size), dtype="complex"))
            self.add_output('circ_wake_' + str(t), val=numpy.zeros((size_wake), dtype="complex"))
            self.add_output('v_wake_on_wing_' + str(t), val=numpy.zeros((size, 3), dtype="complex"))

            self.circ_wake = numpy.zeros((size_wake), dtype="complex")
            self.v_wake = numpy.zeros((size, 3), dtype="complex")
            self.a1_mtx = numpy.zeros((size, size_a1, 3), dtype="complex")

        # from dt=2 the wake circulations of the last dt have to be passed as input,
        # because we need to copy them
        if t > 1:
            self.add_input('circ_wake_' + str(lt), val=numpy.zeros((size_wake - (n-1)), dtype="complex"))

        self.num_x = nx
        self.num_y = n
        self.t = t
        self.lt = lt
        self.iw = iw

        self.mtx = numpy.zeros((size, size), dtype="complex")
        self.rhs = numpy.zeros((size), dtype="complex")
        self.a_mtx = numpy.zeros((size, size, 3), dtype="complex")
        self.b_mtx = numpy.zeros((size, size, 3), dtype="complex")

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd', form='central')

    def compute(self, inputs, outputs):
        nx = self.num_x
        n = self.num_y
        t = self.t
        lt = self.lt
        iw = self.iw

        aic_mtx, _ = assemble_AIC_matrices(outputs['b_pts_' + str(t)], outputs['c_pts_' + str(t)], self.a_mtx, self.b_mtx)

        # Set the wake circulations based on previous circulations from the wing.
        if t == 1:
            # In the first timestep there's only one circulation to pass to the wake
            outputs['circ_wake_' + str(t)] = outputs['circ_' + str(lt)][(nx-2)*(n-1):]

        if t > 1:
            # Set the first row of the wake circulations with the last row of the wing circulations
            outputs['circ_wake_' + str(t)][:(n-1)] = outputs['circ_' + str(lt)][(nx-2)*(n-1):]
            outputs['circ_wake_' + str(t)][(n-1):] = outputs['circ_wake_' + str(lt)]

        # velocity of c_pts (wing panels) induced by wake panels (in inertial frame)
        if t > 0:
            # Shift the c_pts in the negative direction based on the velocity
            # at the wing. This velocity is just freestream velocity.
            c_pts_inertial_frame = outputs['c_pts_' + str(t)] - outputs['v_local_' + str(t)] * outputs['dt'] * t

            # The inertial frame is body-fixed.
            # Therefore we must update the points based on the velocities.

            # a1_mtx is the matrix used to get the induced velocities on the wing from the wake.
            # This uses the wake_mesh which is really the b_pts for the wake
            self.a1_mtx, _ = assemble_AIC_matrices(outputs['wake_mesh_' + str(t)][:iw+1, :, :], c_pts_inertial_frame, self.a1_mtx, wake=True)

            # Dot the matrix with the wake circulations to get the induced velocity
            # on the wing from the wake.
            # This is used to set the RHS of the system to solve for the new circulations.
            for ind in range(3):
                outputs['v_wake_on_wing_' + str(t)][:, ind] = self.a1_mtx[:, :, ind].dot(outputs['circ_wake_' + str(t)][:(n-1)*iw])

        # Set up the matrix used to compute the circulations on the wing.
        # Do this by dotting the matrix with the normals.
        # We do the same thing in OAS currently.
        self.mtx[:, :] = 0.
        for ind in range(3):
            self.mtx[:, :] += (aic_mtx[:, :, ind].T \
                            * outputs['normals_' + str(t)][:, :, ind].flatten('C')).T

        # On the first timestep, only the freestream velocity is acting on the panels.
        v_c_pts = outputs['v_local_' + str(t)].reshape(-1, 3, order='C')

        if t > 0:
            # On subsequent timesteps, both the freestream velocity and the velocity
            # induced by the wake acts on the wing
            v_c_pts += outputs['v_wake_on_wing_' + str(t)]

        # Reshape the normals so that we can correctly produce the rhs
        norm = outputs['normals_' + str(t)].reshape(-1, 3, order='C')

        # Populate the rhs vector
        self.rhs = numpy.sum(-norm * v_c_pts, axis=1)

        # Solve the linear system with the prepared matrix and rhs to get
        # the circulations acting on the wing.
        # Note that this is based on the effects from the wing itself and the wake.
        outputs['circ_' + str(t)] = numpy.linalg.solve(self.mtx, self.rhs)

class InducedVelocities(om.ExplicitComponent):
    """ Define induced velocities acting on each wing (v) or wake (w) panel
    'v_wing_on_wing_' + str(t) = velocity of c_pts (wing panels) induced by wing panels, at the time step t
    'v_wakewing_on_wake_' + str(t) = velocity of wake_mesh (wake points) induced by wing and wake panels, at the time step t """

    def initialize(self):
        self.options.declare("nx")
        self.options.declare("n")
        self.options.declare("t")
        self.options.declare("nw")

    def setup(self):
        nx = self.options["nx"]
        n = self.options["n"]
        t = self.options["t"]
        nw = self.options["nw"]

        if t < nw:
            iw = t
        else:
            iw = nw

        self.add_input('v', val=10.)
        self.add_input('dt', val=0.1)
        self.add_input('b_pts_' + str(t), val=numpy.zeros((nx, n, 3)))
        self.add_input('c_pts_' + str(t), val=numpy.zeros((nx-1, n-1, 3)))
        size = (nx-1) * (n-1)
        self.add_input('circ_' + str(t), val=numpy.zeros((size), dtype="complex"))
        self.add_output('v_wing_on_wing_' + str(t), val=numpy.zeros((size, 3), dtype="complex"))

        size_wake_mesh = n * iw
        size_a3 = (n-1) * iw

        # Only need wake induced velocity if it exists
        if t > 0:
            size_wake = (n-1) * t
            self.add_input('circ_wake_' + str(t), val=numpy.zeros((size_wake), dtype="complex"))
            self.add_input('wake_mesh_' + str(t), val=numpy.zeros((t+1, n, 3), dtype="complex"))
            self.add_output('v_wakewing_on_wake_' + str(t), val=numpy.zeros((size_wake_mesh, 3), dtype="complex"))

            self.w_wake = numpy.zeros((size_wake_mesh, 3), dtype="complex")
            self.a3_mtx = numpy.zeros((size_wake_mesh, size_a3, 3), dtype="complex")

        self.num_y = n
        self.t = t
        self.iw = iw

        self.v_wing = numpy.zeros((size, 3), dtype="complex")
        self.w_wing = numpy.zeros((size_wake_mesh, 3), dtype="complex")
        self.a2_mtx = numpy.zeros((size_wake_mesh, size, 3), dtype="complex")

        # Cap the size of the wake mesh based on the desired number of wake rows
        if t < nw:
            self.wake_mesh_local_frame = numpy.zeros((t+1, n, 3), dtype="complex")
        else:
            self.wake_mesh_local_frame = numpy.zeros((nw+1, n, 3), dtype="complex")

        self.a_mtx = numpy.zeros((size, size, 3), dtype="complex")
        self.b_mtx = numpy.zeros((size, size, 3), dtype="complex")

    def compute(self, inputs, outputs):

        n = self.num_y
        t = self.t
        iw = self.iw

        _, b_AIC_mtx = assemble_AIC_matrices(outputs['b_pts_' + str(t)], outputs['c_pts_' + str(t)], self.a_mtx, self.b_mtx, True)

        # Obtain the induced velocity on the wing caused by the wing
        # by using the b_mtx previously obtained.
        # We currently do this in OAS too.
        for ind in range(3):
            self.v_wing[:, ind] = b_AIC_mtx[:, :, ind].dot(outputs['circ_' + str(t)])

        # Induced velocity on wing caused by wing
        outputs['v_wing_on_wing_' + str(t)] = self.v_wing

        # Wake rollup (w)
        if t > 0:

            # Translate the wake mesh into the local frame
            translation = numpy.array([outputs['v'] * outputs['dt'] * t, 0., 0.])
            self.wake_mesh_local_frame = outputs['wake_mesh_' + str(t)] + translation

            # outputs['wake_mesh_' + str(t)] doesn't change with each timestep
            # but self.wake_mesh_local_frame does change with each timestep

            # Assemble a2_mtx which is used to calculate the wing induced velocity
            # caused by the wake
            self.a2_mtx, _ = assemble_AIC_matrices(outputs['b_pts_' + str(t)], self.wake_mesh_local_frame[1:,:,], self.a2_mtx)

            # Assemble a3_mtx which is used to calculate the wake induced velocity
            # caused by the wake
            self.a3_mtx, _ = assemble_AIC_matrices(outputs['wake_mesh_' + str(t)][:(iw+1), :, :],
                                  outputs['wake_mesh_' + str(t)][1:(iw+1), :, :], self.a3_mtx)

            # Obtain the induced velocities on the wake caused by the wing and the wake
            for ind in range(3):
                self.w_wing[:, ind] = self.a2_mtx[:, :, ind].dot(outputs['circ_' + str(t)])
                self.w_wake[:, ind] = self.a3_mtx[:, :, ind].dot(outputs['circ_wake_' + str(t)][:iw*(n-1)])

            # Induced velocity on the wake caused by wing and wake
            outputs['v_wakewing_on_wake_' + str(t)][:] = self.w_wing + self.w_wake

class WakeGeometry(om.ExplicitComponent):
    """ Update position of wake mesh in the body frame, adding a line for each time step
    'wake_mesh_' + str(nt) = position of wake mesh points (wake rings corners), at the time step t+1 """

    def initialize(self):
        self.options.declare("nx")
        self.options.declare("n")
        self.options.declare("t")
        self.options.declare("nw")

    def setup(self):
        nx = self.options["nx"]
        n = self.options["n"]
        t = self.options["t"]
        nw = self.options["nw"]

        nt = t + 1

        if t < nw:
            iw = t
        else:
            iw = nw

        self.add_input('v', val=10.)
        self.add_input('dt', val=0.1)
        self.add_input('starting_vortex_' + str(t), val=numpy.zeros((1, n, 3), dtype="complex"))
        self.add_output('wake_mesh_' + str(nt), val=numpy.zeros((t+2, n, 3), dtype="complex"))

        size_wake_mesh = n * iw
        if t > 0:
            self.add_input('v_wakewing_on_wake_' + str(t), val=numpy.zeros((size_wake_mesh, 3), dtype="complex"))
            self.add_input('wake_mesh_' + str(t), val=numpy.zeros((t+1, n, 3), dtype="complex"))

        self.num_y = n
        self.t = t
        self.nt = nt
        self.iw = iw

        self.v_wakewing_on_wake_resh = numpy.zeros((iw, n, 3), dtype="complex")
        self.new_wake_row = numpy.zeros((1, n, 3), dtype="complex")
        self.old_wake = numpy.zeros((t+1, n, 3), dtype="complex")

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd', form='central')

    def compute(self, inputs, outputs):

        n = self.num_y
        t = self.t
        nt = self.nt
        iw = self.iw

        # Reshape of v_wakewing_on_wake so it can easily be applied to the wake mesh
        if t > 0:
            for ind in range(3):
                self.v_wakewing_on_wake_resh[:, :, ind] = outputs['v_wakewing_on_wake_' + str(t)][:, ind].reshape(iw, n, order='C')

        # Set old_wake based on if this is the first timestep or subsequent ones
        if t == 0:
            self.old_wake = outputs['starting_vortex_' + str(t)]
        else:
            self.old_wake = outputs['wake_mesh_' + str(t)]

            # Here we apply the reshaped induced velocity on the wake caused by
            # the wing and the wake to the wake mesh.
            # This gives us the wake mesh from the third row to the end.
            outputs['wake_mesh_' + str(nt)][2:(iw+2), :, :] = self.old_wake[1:(iw+1), :, :] \
                                                   + self.v_wakewing_on_wake_resh * outputs['dt']
            outputs['wake_mesh_' + str(nt)][(iw+2):, :, :] = self.old_wake[(iw+1):, :, :]

        # Update the second wake_mesh row
        outputs['wake_mesh_' + str(nt)][1, :, :] = self.old_wake[0, :, :]

        # Addition of a new wake row
        self.new_wake_row = outputs['starting_vortex_' + str(t)]
        self.new_wake_row[0, :, 0] -= outputs['v'] * outputs['dt'] * (t+1)

        # Set the first wake_mesh row based on the starting vortex and the distance
        # traveled since then.
        outputs['wake_mesh_' + str(nt)][0, :, :] = self.new_wake_row

class Forces(om.ExplicitComponent):
    """ Define aerodynamic forces acting on each wing panel,
    evaluated by unsteady Bernoulli formula
    forces_L_t = lift acting on each wing panel, at the time step t
    forces_D_t = induced drag acting on each wing panel, at the time step t """

    def initialize(self):
        self.options.declare("nx")
        self.options.declare("n")
        self.options.declare("t")

    def setup(self):
        nx = self.options["nx"]
        n = self.options["n"]
        t = self.options["t"]

        lt = t - 1

        self.add_input('rho', val=3.)
        self.add_input('alpha', val=3.)
        self.add_input('dt', val=0.1)
        self.add_input('lengths_' + str(t), val=numpy.zeros((nx-1, n-1)))
        self.add_input('widths_' + str(t), val=numpy.zeros((nx-1, n-1)))
        self.add_input('normals_' + str(t), val=numpy.zeros((nx-1, n-1, 3), dtype="complex"))
        self.add_input('v_local_' + str(t), val=numpy.zeros((nx-1, n-1, 3), dtype="complex"))
        size = (nx-1) * (n-1)
        self.add_input('circ_' + str(t), val=numpy.zeros((size), dtype="complex"))
        self.add_input('v_wing_on_wing_' + str(t), val=numpy.zeros((size, 3), dtype="complex"))

        self.add_output('sigma_x_' + str(t), val=numpy.zeros((nx-1, n-1), dtype="complex"))
        self.add_output('sec_L_' + str(t), val=numpy.zeros((nx-1, n-1), dtype="complex"))
        self.add_output('sec_D_' + str(t), val=numpy.zeros((nx-1, n-1), dtype="complex"))
        self.add_output('sec_forces_' + str(t), val=numpy.zeros((n-1, 3), dtype="complex"))

        if t > 0:
            size_wake = (n-1) * t
            self.add_input('v_wake_on_wing_' + str(t), val=numpy.zeros((size, 3), dtype="complex"))
            self.add_input('sigma_x_' + str(lt), val=numpy.zeros((nx-1, n-1), dtype="complex"))

        self.num_y = n
        self.num_x = nx
        self.t = t
        self.lt = lt

        self.velo = numpy.zeros((nx-1, n-1, 3), dtype="complex")
        self.loc_circ = numpy.zeros((nx-1, n-1), dtype="complex")
        self.sum_sections = numpy.zeros((n-1, 3), dtype="complex")

    def compute(self, inputs, outputs):

        n = self.num_y
        nx = self.num_x
        t = self.t
        lt = self.lt

        # If this is the first timestep, only the freestream velocity is acting on the panels
        self.velo = outputs['v_local_' + str(t)]
        self.vind = outputs['v_wing_on_wing_' + str(t)][:, 2].reshape(nx-1, n-1, order='C')

        if t > 0:
            # For subsequent timesteps, the induced velocities are from the freestream and induced
            for ind in range(3):
                self.velo[:, :, ind] += outputs['v_wake_on_wing_' + str(t)][:, ind].reshape(nx-1, n-1, order='C')

            self.vind += outputs['v_wake_on_wing_' + str(t)][:, 2].reshape(nx-1, n-1, order='C')

        # Reshape the circulations into a matrix so we can more easily manipulate the values
        circ_mtx = outputs['circ_' + str(t)].reshape(nx-1, n-1, order='C')

        # For the first row of circulations, use the values.
        # For all other rows, use the difference between that value and the previous value.
        # This is necessary when using vortex rings to get the correct effects.
        self.loc_circ[0, :] = circ_mtx[0, :]
        self.loc_circ[1:, :] = circ_mtx[1:, :] - circ_mtx[:-1, :]

        # Velocity-potential time derivative (dCirc_dt) is obtained by integrating
        # from the leading edge
        outputs['sigma_x_' + str(t)] = 0.5 * self.loc_circ
        outputs['sigma_x_' + str(t)][1:, :] += circ_mtx[:-1, :]
        outputs['sigma_x_' + str(t)] *= outputs['lengths_' + str(t)]

        # Obtain the change in circulation per timestep
        if t == 0:
            dCirc_dt = outputs['sigma_x_' + str(t)] / outputs['dt']
        else:
            dCirc_dt = (outputs['sigma_x_' + str(t)] - outputs['sigma_x_' + str(lt)]) / outputs['dt']

        # Lift for each panel
        forces_L = (self.velo[:, :, 0] * self.loc_circ + dCirc_dt) * outputs['widths_' + str(t)] * outputs['normals_' + str(t)][:, :, 2]

        # Induced drag for each panel
        forces_D = outputs['widths_' + str(t)] * (-self.vind * self.loc_circ
             + dCirc_dt * outputs['normals_' + str(t)][:, :, 0])

        outputs['sec_L_' + str(t)] = forces_L * outputs['rho']
        outputs['sec_D_' + str(t)] = forces_D * outputs['rho']

        print("L: {},  D: {}".format(numpy.sum(outputs['sec_L_' + str(t)]), numpy.sum(outputs['sec_D_' + str(t)])))

        # section forces for structural part
        projected_forces = numpy.array(outputs['normals_' + str(t)], dtype="complex")
        for ind in range(3):
            projected_forces[:, :, ind] *= forces_L

        outputs['sec_forces_' + str(t)] = numpy.zeros((n-1, 3))
        for x in range(nx-1):
            self.sum_sections += projected_forces[x, :, :]
        outputs['sec_forces_' + str(t)] = self.sum_sections

class LiftDrag(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("nx")
        self.options.declare("n")
        self.options.declare("num_dt")

    def setup(self):
        nx = self.options["nx"]
        n = self.options["n"]
        num_dt = self.options["num_dt"]

        self.add_input('sec_L_' + str(num_dt-1), val=numpy.zeros((nx-1, n-1), dtype="complex"))
        self.add_input('sec_D_' + str(num_dt-1), val=numpy.zeros((nx-1, n-1), dtype="complex"))
        self.add_output('L', val=0.)
        self.add_output('D', val=0.)
        self.num_dt = num_dt

    def compute(self, inputs, outputs):
        outputs['L'] = numpy.sum(outputs['sec_L_' + str(self.num_dt - 1)])
        outputs['D'] = numpy.sum(outputs['sec_D_' + str(self.num_dt - 1)])

class AeroCoeffs(om.ExplicitComponent):
    def setup(self):
        self.add_input('S_ref', val=0.)
        self.add_input('rho', val=0.)
        self.add_input('v', val=0.)
        self.add_input('L', val=0.)
        self.add_input('D', val=0.)
        self.add_output('CL1', val=0.)
        self.add_output('CDi', val=0.)

    def compute(self, inputs, outputs):
        S_ref = outputs['S_ref']
        rho = outputs['rho']
        v = outputs['v']
        L = outputs['L']
        D = outputs['D']

        outputs['CL1'] = L / (0.5*rho*v**2*S_ref)
        outputs['CDi'] = D / (0.5*rho*v**2*S_ref)

class TotalLift(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("CL0")

    def setup(self):
        CL0 = self.options["CL0"]

        self.add_input('CL1', val=1.)
        self.add_output('CL', val=1.)
        self.CL0 = CL0

    def compute(self, inputs, outputs):
        outputs['CL'] = outputs['CL1'] + self.CL0

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd', form='central')

class TotalDrag(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("CD0")

    def setup(self):
        CD0 = self.options["CD0"]

        self.add_input('CDi', val=1.)
        self.add_output('CD', val=1.)
        self.CD0 = CD0

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd', form='central')

    def compute(self, inputs, outputs):
        outputs['CD'] = outputs['CDi'] + self.CD0

class UVLMStates(om.Group):
    """ Group that contains the aerodynamic states """

    def __init__(self, num_x, num_y, num_w, t):
        super(UVLMStates, self).__init__()

        self.add_subsystem('wgeom_%d'%t,
                 Geometry(nx=num_x, n=num_y, t=t),
                 promotes=['*'])
        self.add_subsystem('circ_%d'%t,
                 Circulations(nx=num_x, n=num_y, nw=num_w, t=t),
                 promotes=['*'])
        self.add_subsystem('indvel_%d'%t,
                 InducedVelocities(nx=num_x, n=num_y, nw=num_w, t=t),
                 promotes=['*'])
        self.add_subsystem('wakegeom_%d'%t,
                 WakeGeometry(nx=num_x, n=num_y, nw=num_w, t=t),
                 promotes=['*'])
        self.add_subsystem('forces_%d'%t,
                 Forces(nx=num_x, n=num_y, t=t),
                 promotes=['*'])


class UVLMFunctionals(om.Group):
    """ Group that contains the aerodynamic functionals used to evaluate performance """

    def __init__(self, num_x, num_y, CL0, CD0, num_dt):
        super(UVLMFunctionals, self).__init__()

        self.add_subsystem('liftdrag',
                 LiftDrag(nx=num_x, n=num_y, num_dt=num_dt),
                 promotes=['*'])
        self.add_subsystem('aero_coeffs',
             AeroCoeffs(),
                 promotes=['*'])
        self.add_subsystem('total_CL',
                 TotalLift(CL0=CL0),
                 promotes=['*'])
        self.add_subsystem('total_CD',
                 TotalDrag(CD0=CD0),
                 promotes=['*'])
