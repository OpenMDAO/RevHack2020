import unittest

import cma

import numpy as np
import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal

class CMAESTestCase(unittest.TestCase):

    def test_rosenbrock_om(self):
        #
        # test case from test_differential_evolution_driver
        #

        ORDER = 6  # dimension of problem

        span = 2   # upper and lower limits
        lower_bound = -span*np.ones(ORDER)
        upper_bound = span*np.ones(ORDER)

        class RosenbrockComp(om.ExplicitComponent):
            """
            nth dimensional Rosenbrock function, array input and scalar output
            global minimum at f(1,1,1...) = 0
            """
            def initialize(self):
                self.options.declare('order', types=int, default=2, desc='dimension of input.')

            def setup(self):
                self.add_input('x', np.zeros(self.options['order']))
                self.add_output('y', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                n = len(x)
                assert (n > 1)
                s = 0
                for i in range(n - 1):
                    s += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2

                outputs['y'] = s

        rosenbrock_model = om.Group()
        rosenbrock_model.add_subsystem('rosenbrock', RosenbrockComp(order=ORDER))
        rosenbrock_model.add_design_var('rosenbrock.x', lower=lower_bound, upper=upper_bound)
        rosenbrock_model.add_objective('rosenbrock.y')

        p = om.Problem(model=rosenbrock_model, driver=om.DifferentialEvolutionDriver(max_gen=800))
        p.setup()
        p.run_driver()

        assert_near_equal(p['rosenbrock.y'], 0.0, 1e-3)
        assert_near_equal(p['rosenbrock.x'], np.ones(ORDER), 1e-3)

    def test_rosenbrock_cma(self):
        #
        # test case from cma.test
        #
        rosenbrock = cma.ff.rosen

        ORDER = 6  # dimension of problem

        span = 2   # upper and lower limits
        lower_bound = -span*np.ones(ORDER)
        upper_bound = span*np.ones(ORDER)

        res = cma.fmin(rosenbrock, [-1]*ORDER, 0.01,
                       options={'ftarget':1e-6, 'bounds':[lower_bound, upper_bound],
                                'verb_time':0, 'verb_disp':0, 'seed':3})

        # - res[0]  (xopt) -- best evaluated solution
        # - res[1]  (fopt) -- respective function value
        # - res[2]  (evalsopt) -- respective number of function evaluations
        # - res[3]  (evals) -- number of overall conducted objective function evaluations
        # - res[4]  (iterations) -- number of overall conducted iterations
        # - res[5]  (xmean) -- mean of the final sample distribution
        # - res[6]  (stds) -- effective stds of the final sample distribution
        # - res[-3] (stop) -- termination condition(s) in a dictionary
        # - res[-2] (cmaes) -- class `CMAEvolutionStrategy` instance
        # - res[-1] (logger) -- class `CMADataLogger` instance

        xopt = res[0]
        fopt = res[1]

        assert_near_equal(fopt, 0.0, 1e-3)
        assert_near_equal(xopt, np.ones(ORDER), 1e-3)

        es = cma.CMAEvolutionStrategy([1]*ORDER, 1).optimize(rosenbrock)

        assert_near_equal(es.result.fbest, 0.0, 1e-3)
        assert_near_equal(es.result.xbest, np.ones(ORDER), 1e-3)

    def test_trust_constr(self):
        #
        # test case from test_scipy_optimizer
        #

        rosenbrock_size = 6  # size of the design variable

        def rosenbrock(x):
            x_0 = x[:-1]
            x_1 = x[1:]
            return sum((1 - x_0) ** 2) + 100 * sum((x_1 - x_0 ** 2) ** 2)

        class Rosenbrock(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.ones(rosenbrock_size))
                self.add_output('f', 0.0)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rosenbrock(x)

        class Rosenbrock(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.array([1.5, 1.5, 1.5]))
                self.add_output('f', 0.0)
                self.declare_partials('f', 'x', method='fd', form='central', step=1e-2)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rosenbrock(x)

        x0 = np.array([1.2, 0.8, 1.3])

        prob = om.Problem()
        model = prob.model
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', list(x0))

        prob.model.add_subsystem('rosen', Rosenbrock(), promotes=['*'])
        prob.model.add_subsystem('con', om.ExecComp('c=sum(x)', x=np.ones(3)), promotes=['*'])
        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'trust-constr'
        driver.options['tol'] = 1e-8
        driver.options['maxiter'] = 2000
        driver.options['disp'] = False

        model.add_design_var('x')
        model.add_objective('f', scaler=1/rosenbrock(x0))
        model.add_constraint('c', lower=0, upper=10)  # Double sided

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['x'], np.ones(3), 2e-2)
        assert_near_equal(prob['f'], 0., 1e-2)
        self.assertTrue(prob['c'] < 10)
        self.assertTrue(prob['c'] > 0)

class CMAESDriverTestCase(unittest.TestCase):

    # Shamelessly copied first of openmdao/drivers/tests/test_genetic_algorithm_driver.py
    # max_gen and pop_size options should be available with CMAES as well
    def test_simple_test_func(self):
        from cmaes_driver import CMAESDriver

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

