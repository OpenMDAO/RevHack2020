import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from evtol_dynamics_comp import CLfunc
from evtol_dynamics_comp_vectorized import CLfunc as cl_func_vectorized

import verify_data

class TestCL(unittest.TestCase):

    def test_cl_func_vectorized(self):

        alpha_stall = 0.26179939
        AR = 8
        e = 0.68
        a0 = 5.9
        t_over_c = 0.12

        aoa_blown = np.array([-0.07188392, 0.2270590416478, 0.2364583821384856, 0.2401759902150005, 0.2423804104464628, 0.243943990853836])

        CL = cl_func_vectorized(aoa_blown, alpha_stall, AR, e, a0, t_over_c)

        expected = np.array([-0.3152743051485073, 0.9958538389888826, 1.0370761798060673,
                              1.0533747112777043, 1.063032129491213, 1.069874574395663])

        with np.printoptions(precision=16):
            print(CL)

        assert_near_equal(CL, expected, tolerance=1.0E-8)

    def test_cl_func(self):
        CL = []

        for aoa_blown in [-0.07188392, 0.2270590416478, 0.2364583821384856, 0.2401759902150005, 0.2423804104464628, 0.243943990853836]:
            alpha_stall = 0.26179939
            AR = 8
            e = 0.68
            a0 = 5.9
            t_over_c = 0.12

            CL.append(CLfunc(aoa_blown, alpha_stall, AR, e, a0, t_over_c))


        expected = np.array([-0.3152743051485073, 0.9958538389888826, 1.0370761798060673,
                              1.0533747112777043, 1.063032129491213, 1.069874574395663])

        with np.printoptions(precision=16):
            print(CL)

        assert_near_equal(CL, expected, tolerance=1.0E-8)

if __name__ == "__main__":
    unittest.main()