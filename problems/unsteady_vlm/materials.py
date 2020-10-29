import numpy

import openmdao.api as om

class MaterialsTube(om.ExplicitComponent):
    """ Computes geometric properties for a tube element """

    def __init__(self, n):
        super(MaterialsTube, self).__init__()

        self.add_input('r', val=numpy.zeros((n - 1)))
        self.add_input('thick', val=numpy.zeros((n - 1)))
        self.add_output('A', val=numpy.zeros((n - 1)))
        self.add_output('Iy', val=numpy.zeros((n - 1)))
        self.add_output('Iz', val=numpy.zeros((n - 1)))
        self.add_output('J', val=numpy.zeros((n - 1)))


        self.declare_partials(of='*', wrt='*', method='fd', form='central')
        self.arange = numpy.arange(n-1)

    def solve_nonlinear(self, params, unknowns, resids):

        pi = numpy.pi
        r1 = params['r'] - 0.5 * params['thick']
        r2 = params['r'] + 0.5 * params['thick']

        unknowns['A'] = pi * (r2**2 - r1**2)
        unknowns['Iy'] = pi * (r2**4 - r1**4) / 4.
        unknowns['Iz'] = pi * (r2**4 - r1**4) / 4.
        unknowns['J'] = pi * (r2**4 - r1**4) / 2.

    def compute_partials(self, params, unknowns, resids):
        pi = numpy.pi
        r = params['r'].real
        t = params['thick'].real
        r1 = r - 0.5 * t
        r2 = r + 0.5 * t

        dr1_dr = 1.
        dr2_dr = 1.
        dr1_dt = -0.5
        dr2_dt =  0.5

        r1_3 = r1**3
        r2_3 = r2**3

        a = self.arange
        unknowns['A', 'r'][a, a] = 2 * pi * (r2 * dr2_dr - r1 * dr1_dr)
        unknowns['A', 'thick'][a, a] = 2 * pi * (r2 * dr2_dt - r1 * dr1_dt)
        unknowns['Iy', 'r'][a, a] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        unknowns['Iy', 'thick'][a, a] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        unknowns['Iz', 'r'][a, a] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        unknowns['Iz', 'thick'][a, a] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        unknowns['J', 'r'][a, a] = 2 * pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        unknowns['J', 'thick'][a, a] = 2 * pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)

        return unknowns


