""" Defines the structural analysis component using spatial beam theory """

import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

from pyNBSolver import pyNBSolver
import openmdao.api as om
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import gmres, LinearOperator, splu
import scipy.sparse as sp

def view_mat(mat):
    import matplotlib.pyplot as plt
    if len(mat.shape) > 2:
        mat = numpy.sum(mat, axis=2)
    print("Cond #:", numpy.linalg.cond(mat))
    im = plt.imshow(mat.real, interpolation='none')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

def norm(vec):
    return numpy.sqrt(numpy.sum(vec**2))

def unit(vec):
    return vec / norm(vec)

def radii(mesh, t_c=0.15):
    vectors = mesh[-1, :, :] - mesh[0, :, :]
    chords = numpy.sqrt(numpy.sum(vectors**2, axis=1))
    chords = 0.5 * chords[:-1] + 0.5 * chords[1:]
    return t_c * chords

def order(evalues, evectors):
    idx = numpy.imag(evalues).argsort()[::1]
    eigenvalues = evalues[idx]
    eigenvectors = evectors[:,idx]
    return eigenvalues, eigenvectors

def Eig_matrix(M_array, K_array):
    M = numpy.asmatrix(M_array)
    K = numpy.asmatrix(K_array)
    M_inv = numpy.linalg.inv(M)
    return M_inv*K

def assemble_system(A, Iy, Iz, mesh, J, E, G, mrho, fem_origin,
                    elem_IDs, x_gl, T_elem, T, S_a, S_t, S_y, S_z,
                    const1, const_y, const_z, kk_a, kk_t, kk_y, kk_z, K_elem, K_mtx,
                    const2, const_yy, const_zz, mm_a, mm_t, mm_y, mm_z, M_elem, M_mtx):

    w = fem_origin
    nodes = (1-w) * mesh[0, :, :] + w * mesh[-1, :, :]

    num_elems = elem_IDs.shape[0]
    num_nodes = nodes.shape[0]

    elem_nodes = numpy.zeros((num_elems, 2, 3), dtype='complex')

    for ielem in range(num_elems):
        in0, in1 = elem_IDs[ielem, :]
        elem_nodes[ielem, 0, :] = nodes[in0, :]
        elem_nodes[ielem, 1, :] = nodes[in1, :]

    E, G = E * numpy.ones(num_nodes - 1), G * numpy.ones(num_nodes - 1)

    K_mtx[:] = 0.
    M_mtx[:] = 0.
    '''
    print "A =", A[ielem]
    print "J =", J
    print "Iy =", Iy
    print "Iz =", Iz
    print "mrho =", mrho
    print "E =", E
    print
    '''
    J2 = J
    #J2[:] = 6.361725

    for ielem in range(num_elems):
        P0 = elem_nodes[ielem, 0, :]
        P1 = elem_nodes[ielem, 1, :]

        x_loc = unit(P1 - P0)
        y_loc = unit(numpy.cross(x_loc, x_gl))
        z_loc = unit(numpy.cross(x_loc, y_loc))

        T[0, :] = x_loc
        T[1, :] = y_loc
        T[2, :] = z_loc
        for ind in range(4):
            T_elem[3*ind:3*ind+3, 3*ind:3*ind+3] = T

        ####################
        # Stiffness matrix #
        ####################
        L = norm(P1 - P0)
        EA_L = E[ielem] * A[ielem] / L
        GJ_L = G[ielem] * J2[ielem] / L
        GJ_L = 10000 / L
        EIy_L3 = E[ielem] * Iy[ielem] / L**3
        EIy_L3 = 2.e4 / L**3
        EIz_L3 = E[ielem] * Iz[ielem] / L**3
        EIz_L3 = 4.e6 / L**3

        kk_a[:, :] = EA_L * const1
        kk_t[:, :] = GJ_L * const1

        kk_y[:, :] = EIy_L3 * const_y
        kk_y[1, :] *= L
        kk_y[3, :] *= L
        kk_y[:, 1] *= L
        kk_y[:, 3] *= L

        kk_z[:, :] = EIz_L3 * const_z
        kk_z[1, :] *= L
        kk_z[3, :] *= L
        kk_z[:, 1] *= L
        kk_z[:, 3] *= L

        K_elem[:] = 0
        K_elem += S_a.T.dot(kk_a).dot(S_a)
        K_elem += S_t.T.dot(kk_t).dot(S_t)
        K_elem += S_y.T.dot(kk_y).dot(S_y)
        K_elem += S_z.T.dot(kk_z).dot(S_z)

        res = T_elem.T.dot(K_elem).dot(T_elem)

        in0, in1 = elem_IDs[ielem, :]
        K_mtx[6*in0:6*in0+6, 6*in0:6*in0+6] += res[:6, :6]
        K_mtx[6*in1:6*in1+6, 6*in0:6*in0+6] += res[6:, :6]
        K_mtx[6*in0:6*in0+6, 6*in1:6*in1+6] += res[:6, 6:]
        K_mtx[6*in1:6*in1+6, 6*in1:6*in1+6] += res[6:, 6:]

        ###############
        # Mass matrix #
        ###############
        mrhoAL = mrho * A[ielem] * L
        mrhoAL = 0.75 * L
        mrhoJL = mrho * J[ielem] * L    # (J = Iy + Iz)
        mrhoJL = 0.047 * L  # (J = Iy + Iz)
        mm_a[:, :] = mrhoAL * const2
        mm_t[:, :] = mrhoJL * const2

        mm_y[:, :] = mrhoAL * const_yy /420
        mm_y[1, :] *= L
        mm_y[3, :] *= L
        mm_y[:, 1] *= L
        mm_y[:, 3] *= L

        mm_z[:, :] = mrhoAL * const_zz /420
        mm_z[1, :] *= L
        mm_z[3, :] *= L
        mm_z[:, 1] *= L
        mm_z[:, 3] *= L

        M_elem[:] = 0
        M_elem += S_a.T.dot(mm_a).dot(S_a)
        M_elem += S_t.T.dot(mm_t).dot(S_t)
        M_elem += S_y.T.dot(mm_y).dot(S_y)
        M_elem += S_z.T.dot(mm_z).dot(S_z)

        res_mass = T_elem.T.dot(M_elem).dot(T_elem)

        M_mtx[6*in0:6*in0+6, 6*in0:6*in0+6] += res_mass[:6, :6]
        M_mtx[6*in1:6*in1+6, 6*in0:6*in0+6] += res_mass[6:, :6]
        M_mtx[6*in0:6*in0+6, 6*in1:6*in1+6] += res_mass[:6, 6:]
        M_mtx[6*in1:6*in1+6, 6*in1:6*in1+6] += res_mass[6:, 6:]


class SpatialBeamMatrices(om.ExplicitComponent):
    """ Computes matrices for dynamic integration """

    def __init__(self, nx, n, E, G, mrho, fem_origin=0.35):
        super(SpatialBeamMatrices, self).__init__()

        self.add_input('zeta', val=0.)
        self.add_input('A', val=numpy.zeros((n - 1)))
        self.add_input('Iy', val=numpy.zeros((n - 1)))
        self.add_input('Iz', val=numpy.zeros((n - 1)))
        self.add_input('J', val=numpy.zeros((n - 1)))
        self.add_input('mesh', val=numpy.zeros((nx, n, 3), dtype="complex"))

        self.size = size = 6 * n
        self.add_output('M_matrix', val=numpy.zeros((size, size), dtype="complex"))
        self.add_output('K_matrix', val=numpy.zeros((size, size), dtype="complex"))

        self.n = n
        self.E = E
        self.G = G
        self.mrho = mrho
        self.fem_origin = fem_origin

        ########################
        # Structural mesh data #
        ########################
        elem_IDs = numpy.zeros((n-1, 2), int)
        elem_IDs[:, 0] = numpy.arange(n-1)
        elem_IDs[:, 1] = numpy.arange(n-1) + 1
        self.elem_IDs = elem_IDs
        self.x_gl = numpy.array([1, 0, 0], dtype='complex')
        self.T_elem = numpy.zeros((12, 12), dtype='complex')
        self.T = numpy.zeros((3, 3), dtype='complex')

        num_nodes = n

        self.S_a = numpy.zeros((2, 12), dtype='complex')
        self.S_a[(0, 1),
                 (0, 6)] = 1.
        self.S_t = numpy.zeros((2, 12), dtype='complex')
        self.S_t[(0, 1),
                 (3, 9)] = 1.
        self.S_y = numpy.zeros((4, 12), dtype='complex')
        self.S_y[(0, 1, 2,  3),
                 (2, 4, 8, 10)] = 1.
        self.S_z = numpy.zeros((4, 12), dtype='complex')
        self.S_z[(0, 1, 2,  3),
                 (1, 5, 7, 11)] = 1.

        ####################
        # Stiffness Matrix #
        ####################
        self.const1 = numpy.array([[ 1,-1],
                                   [-1, 1],], dtype='complex')
        self.const_y = numpy.array([[ 12, -6,-12, -6],
                                    [ -6,  4,  6,  2],
                                    [-12,  6, 12,  6],
                                    [ -6,  2,  6,  4],], dtype='complex')
        self.const_z = numpy.array([[ 12,  6,-12,  6],
                                    [  6,  4, -6,  2],
                                    [-12, -6, 12, -6],
                                    [  6,  2, -6,  4],], dtype='complex')

        self.kk_a = numpy.zeros((2, 2), dtype='complex')
        self.kk_t = numpy.zeros((2, 2), dtype='complex')
        self.kk_y = numpy.zeros((4, 4), dtype='complex')
        self.kk_z = numpy.zeros((4, 4), dtype='complex')

        self.K_elem = numpy.zeros((12, 12), dtype='complex')
        self.K_mtx = numpy.zeros((size, size), dtype='complex')

        ###############
        # Mass Matrix #
        ###############
        self.const2 = numpy.array([[1/3, 1/6],
                                   [1/6, 1/3],], dtype='complex')
        self.const_yy = numpy.array([[156,-22, 54, 13],
                                     [-22,  4,-13, -3],
                                     [ 54,-13,156, 22],
                                     [ 13, -3, 22,  4],], dtype='complex')
        self.const_zz = numpy.array([[156, 22, 54,-13],
                                     [ 22,  4, 13, -3],
                                     [ 54, 13,156,-22],
                                     [-13, -3,-22,  4],], dtype='complex')

        self.mm_a = numpy.zeros((2, 2), dtype='complex')
        self.mm_t = numpy.zeros((2, 2), dtype='complex')
        self.mm_y = numpy.zeros((4, 4), dtype='complex')
        self.mm_z = numpy.zeros((4, 4), dtype='complex')

        self.M_elem = numpy.zeros((12, 12), dtype='complex')
        self.M_mtx = numpy.zeros((size, size), dtype='complex')

        self.C_mtx = numpy.zeros((size, size), dtype='complex')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def solve_nonlinear(self, params, unknowns, resids):

        size = self.size

        assemble_system(params['A'], params['Iy'], params['Iz'], params['mesh'], params['J'],
                        self.E, self.G, self.mrho, self.fem_origin,
                        self.elem_IDs, self.x_gl, self.T_elem, self.T,
                        self.S_a, self.S_t, self.S_y, self.S_z,
                        self.const1, self.const_y, self.const_z,
                        self.kk_a, self.kk_t, self.kk_y, self.kk_z, self.K_elem, self.K_mtx,
                        self.const2, self.const_yy, self.const_zz,
                        self.mm_a, self.mm_t, self.mm_y, self.mm_z, self.M_elem, self.M_mtx)

        unknowns['M_matrix'] = self.M_mtx
        unknowns['K_matrix'] = self.K_mtx

    # def linearize(self, params, unknowns, resids):
    #     """ Jacobian for structural matrices """

    #     jac = self.alloc_jacobian()

    #     fd_jac = self.complex_step_jacobian(params, unknowns, resids,
    #                             fd_params=['A','Iy','Iz','J','mesh'],
    #                             fd_unknowns=['M_matrix', 'K_matrix'],
    #                             fd_states=[])

    #     jac.update(fd_jac)
    #     return jac


class SpatialBeamEIG(om.ExplicitComponent):
    """ Computes eigenvalues and eigenvectors. """

    def __init__(self, n, num_dt, final_t):
        super(SpatialBeamEIG, self).__init__()

        self.size = size = 6 * n
        self.size_eig = size_eig = size - 6

        self.add_input('v', val=10.)
        self.add_input('span', val=58.7630524)
        self.add_input('M_matrix', val=numpy.zeros((size, size), dtype="complex"))
        self.add_input('K_matrix', val=numpy.zeros((size, size), dtype="complex"))
        self.add_output('dt', val=0.001)

        self.omega_list = numpy.zeros(4, dtype='complex')
        self.num_dt = num_dt
        self.final_t = final_t

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def createNBSolver(self, params, unknowns):

        self.NBSolver = pyNBSolver(M = self.reduced_M,        # Mass matrix
                                   K = self.reduced_K,        # Stiffness matrix
                                   N = self.num_dt+1,         # Number of timesteps you want to run
                                   dt = unknowns['dt'])       # Timestep

    def solve_nonlinear(self, params, unknowns, resids):

        # NOTE: here this is only because Gio was running it with symmetry; need to check cons here to see which part of the matrix to use
        self.reduced_M = params['M_matrix'][6:, 6:]
        self.reduced_K = params['K_matrix'][6:, 6:]

        unknowns['dt'] = self.final_t / self.num_dt

        self.createNBSolver(params, unknowns)

    # def linearize(self, params, unknowns, resids):
    #     """ Jacobian for eigenvalues and eigenvectors """

    #     jac = self.alloc_jacobian()
    #     fd_jac = self.complex_step_jacobian(params, unknowns, resids,
    #                             fd_params=['v', 'span', 'M_matrix', 'K_matrix'],
    #                             fd_unknowns=['evalues', 'evectors', 'dt'],
    #                             fd_states=[])
    #     jac.update(fd_jac)

    #     return jac


class SpatialBeamFEM(om.ExplicitComponent):
    """ Computes displacements of beam nodes by integrating dynamic system """

    def __init__(self, nx, n, SBEIG, t):
        super(SpatialBeamFEM, self).__init__()

        def_mesh_t = 'def_mesh_%d'%t
        loads_t = 'loads_%d'%t
        disp_aug_t = 'disp_aug_%d'%t

        self.size = size = 6 * n
        self.add_input('dt', val=0.1)
        self.add_input(def_mesh_t, val=numpy.zeros((nx, n, 3)))
        self.add_input('K_matrix', val=numpy.zeros((size, size)))

        self.add_input('loads', val=numpy.zeros((n, 6)))
        self.add_output(disp_aug_t, val=numpy.zeros((size)))

        self.num_y = n
        self.t = t
        self.SBEIG = SBEIG

        self.rhs = numpy.zeros(size, dtype='complex')

        self.def_mesh_t = def_mesh_t
        self.loads_t = loads_t
        self.disp_aug_t = disp_aug_t

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')


    def solve_nonlinear(self, params, unknowns, resids):

        n = self.num_y
        t = self.t
        def_mesh_t = self.def_mesh_t
        loads_t = self.loads_t
        disp_aug_t = self.disp_aug_t

        self.rhs = params['loads'].reshape((6*n))

        self.SBEIG.NBSolver.stepCounter()
        self.SBEIG.NBSolver.setForces(self.rhs[6:])
        self.SBEIG.NBSolver.timeStepping()
        unknowns[disp_aug_t][6:] = self.SBEIG.NBSolver.getCurDispl()

    # def linearize(self, params, unknowns, resids):
    #     """ Jacobian for eigenvalues and eigenvectors """

    #     def_mesh_t = self.def_mesh_t
    #     loads_t = self.loads_t
    #     disp_aug_t = self.disp_aug_t

    #     jac = self.alloc_jacobian()

    #     fd_jac = self.complex_step_jacobian(params, unknowns, resids,
    #                             fd_params=['dt', def_mesh_t, loads_t],
    #                             fd_unknowns=[disp_aug_t],
    #                             fd_states=[])
    #     jac.update(fd_jac)
    #     return jac


class SpatialBeamDisp(om.ExplicitComponent):
    """ Selects displacements from augmented vector """

    def __init__(self, n, t):
        super(SpatialBeamDisp, self).__init__()

        disp_aug_t = 'disp_aug_%d'%t
        disp_t = 'disp_%d'%t

        size = 6 * n
        self.add_input(disp_aug_t, val=numpy.zeros((size)))
        self.add_output(disp_t, val=numpy.zeros((n, 6)))

        self.declare_partials(of='*', wrt='*', method='fd', form='central')

        self.n = n
        self.t = t

        self.disp_aug_t = disp_aug_t
        self.disp_t = disp_t

    def solve_nonlinear(self, params, unknowns, resids):
        n = self.n
        disp_aug_t = self.disp_aug_t
        disp_t = self.disp_t

        unknowns[disp_t] = numpy.array(params[disp_aug_t][:6*n].reshape((n, 6)))

    # def linearize(self, params, unknowns, resids):

    #     disp_aug_t = self.disp_aug_t
    #     disp_t = self.disp_t

    #     jac = self.alloc_jacobian()

    #     arange = self.arange
    #     jac[disp_t, disp_aug_t][arange, arange] = 1.
    #     return jac


class SpatialBeamStates(om.Group):

    def __init__(self, num_x, num_y, E, G, mrho, SBEIG, t):
        super(SpatialBeamStates, self).__init__()

        name_fem_t = 'fem_%d'%t
        name_disp_t = 'disp_%d'%t


        self.add_subsystem(name_fem_t,
                 SpatialBeamFEM(num_x, num_y, SBEIG, t),
                 promotes=['*'])

        self.add_subsystem(name_disp_t,
                 SpatialBeamDisp(num_y, t),
                 promotes=['*'])


class SpatialBeamFunctionals(om.Group):

    def __init__(self, num_y, E, G, stress, mrho):
        super(SpatialBeamFunctionals, self).__init__()

        self.add_subsystem('energy',
                 SpatialBeamEnergy(num_y),
                 promotes=['*'])
        self.add_subsystem('weight',
                 SpatialBeamWeight(num_y, mrho),
                 promotes=['*'])
        self.add_subsystem('vonmises',
                 SpatialBeamVonMisesTube(num_y, E, G),
                 promotes=['*'])
        self.add_subsystem('failure',
                 SpatialBeamFailureKS(num_y, stress),
                 promotes=['*'])
