import unittest 

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from cantilever_beam import BeamMoment


class AnalysisTest(unittest.TestCase): 

    def setUp(self): 
        self.p = p = om.Problem()

        N = 11

        p.model.add_subsystem('beam', BeamMoment(n_q=N, idx_F=7), promotes=['*'])

        p.setup()

    def test_no_point_load(self): 

        p = self.p

        p['F'] = 0
        p['alpha'] = 0
        p['Q'] = 2
        p['l'] = 10.
     
        p.run_model()

        assert_near_equal(p['x'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert_near_equal(p['M'], -p['x'][::-1]**2)

    def test_no_distrib_load(self): 

        p = self.p

        p['F'] = 2
        p['alpha'] = 0
        p['Q'] = 0
        p['l'] = 10.
     
        p.run_model()

        assert_near_equal(p['x'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert_near_equal(p['M'], [14, 12, 10, 8, 6, 4, 2, 0, 0, 0, 0])


    def test_both_loads(self): 

        p = self.p


        p['F'] = 2
        p['alpha'] = 0
        p['Q'] = 2
        p['l'] = 10.
     
        p.run_model()

        assert_near_equal(p['x'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert_near_equal(p['M'], [14, 12, 10, 8, 6, 4, 2, 0, 0, 0, 0] - p['x'][::-1]**2 )


if __name__ == "__main__": 
    unittest.main()

