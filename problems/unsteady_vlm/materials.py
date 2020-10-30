import numpy

import openmdao.api as om

class MaterialsTube(om.ExplicitComponent):
    """ Computes geometric properties for a tube element """

    def initialize(self):
        self.options.declare('n')

    def setup(self):
        n = self.options['n']

        self.add_input('r', val=numpy.zeros((n - 1)))
        self.add_input('thick', val=numpy.zeros((n - 1)))
        self.add_output('A', val=numpy.zeros((n - 1)))
        self.add_output('Iy', val=numpy.zeros((n - 1)))
        self.add_output('Iz', val=numpy.zeros((n - 1)))
        self.add_output('J', val=numpy.zeros((n - 1)))

        self.arange = numpy.arange(n-1)

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*', method='fd', form='central')

    def compute(self, inputs, outputs):

        pi = numpy.pi
        r1 = inputs['r'] - 0.5 * inputs['thick']
        r2 = inputs['r'] + 0.5 * inputs['thick']

        outputs['A'] = pi * (r2**2 - r1**2)
        outputs['Iy'] = pi * (r2**4 - r1**4) / 4.
        outputs['Iz'] = pi * (r2**4 - r1**4) / 4.
        outputs['J'] = pi * (r2**4 - r1**4) / 2.

    def compute_partials(self, inputs, outputs):
        pi = numpy.pi
        r = inputs['r'].real
        t = inputs['thick'].real
        r1 = r - 0.5 * t
        r2 = r + 0.5 * t

        dr1_dr = 1.
        dr2_dr = 1.
        dr1_dt = -0.5
        dr2_dt =  0.5

        r1_3 = r1**3
        r2_3 = r2**3

        a = self.arange
        outputs['A', 'r'][a, a] = 2 * pi * (r2 * dr2_dr - r1 * dr1_dr)
        outputs['A', 'thick'][a, a] = 2 * pi * (r2 * dr2_dt - r1 * dr1_dt)
        outputs['Iy', 'r'][a, a] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        outputs['Iy', 'thick'][a, a] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        outputs['Iz', 'r'][a, a] = pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        outputs['Iz', 'thick'][a, a] = pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)
        outputs['J', 'r'][a, a] = 2 * pi * (r2_3 * dr2_dr - r1_3 * dr1_dr)
        outputs['J', 'thick'][a, a] = 2 * pi * (r2_3 * dr2_dt - r1_3 * dr1_dt)

        return outputs


