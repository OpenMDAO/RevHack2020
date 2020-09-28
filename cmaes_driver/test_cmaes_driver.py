import unittest 

import numpy as np

from cmaes_driver import CMAESDriver

class CMAESDriverTest(unittest.TestCase): 

    # Shamelessly copied first three tests of openmdao/drivers/tests/test_genetic_algorithm_driver.py
    # max_gen and pop_size options should be available with CMAES as well
    def test_simple_test_func(self):
        class MyComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.zeros((2, )))

                self.add_output('a', 0.0)
                self.add_output('b', 0.0)
                self.add_output('c', 0.0)
                self.add_output('d', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                outputs['a'] = (2.0*x[0] - 3.0*x[1])**2
                outputs['b'] = 18.0 - 32.0*x[0] + 12.0*x[0]**2 + 48.0*x[1] - 36.0*x[0]*x[1] + 27.0*x[1]**2
                outputs['c'] = (x[0] + x[1] + 1.0)**2
                outputs['d'] = 19.0 - 14.0*x[0] + 3.0*x[0]**2 - 14.0*x[1] + 6.0*x[0]*x[1] + 3.0*x[1]**2

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', np.array([0.2, -0.2])))
        model.add_subsystem('comp', MyComp())
        model.add_subsystem('obj', om.ExecComp('f=(30 + a*b)*(1 + c*d)'))

        model.connect('px.x', 'comp.x')
        model.connect('comp.a', 'obj.a')
        model.connect('comp.b', 'obj.b')
        model.connect('comp.c', 'obj.c')
        model.connect('comp.d', 'obj.d')

        # Played with bounds so we don't get subtractive cancellation of tiny numbers.
        model.add_design_var('px.x', lower=np.array([0.2, -1.0]), upper=np.array([1.0, -0.2]))
        model.add_objective('obj.f')

        prob.driver = CMAESDriver()
        prob.driver.options['max_gen'] = 75

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('obj.f', prob['obj.f'])
            print('px.x', prob['px.x'])

        # TODO: Satadru listed this solution, but I get a way better one.
        # Solution: xopt = [0.2857, -0.8571], fopt = 23.2933
        assert_near_equal(prob['obj.f'], 12.37306086, 1e-4)
        assert_near_equal(prob['px.x'][0], 0.2, 1e-4)
        assert_near_equal(prob['px.x'][1], -0.88653391, 1e-4)

    def test_mixed_integer_branin(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('xC', 7.5)
        model.set_input_defaults('xI', 0.0)

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = CMAESDriver(max_gen=75, pop_size=25)

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('comp.f', prob['comp.f'])

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49399549, 1e-4)
        self.assertTrue(int(prob['xI']) in [3, -3])

    def test_mixed_integer_branin_discrete(self):
        prob = om.Problem()
        model = prob.model

        indep = om.IndepVarComp()
        indep.add_output('xC', val=7.5)
        indep.add_discrete_output('xI', val=0)

        model.add_subsystem('p', indep)
        model.add_subsystem('comp', BraninDiscrete())

        model.connect('p.xI', 'comp.x0')
        model.connect('p.xC', 'comp.x1')

        model.add_design_var('p.xI', lower=-5, upper=10)
        model.add_design_var('p.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = CMAESDriver(max_gen=75, pop_size=25)

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('comp.f', prob['comp.f'])
            print('p.xI', prob['p.xI'])

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49399549, 1e-4)
        self.assertTrue(prob['p.xI'] in [3, -3])
        self.assertTrue(isinstance(prob['p.xI'], int))