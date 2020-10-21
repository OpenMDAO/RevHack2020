from __future__ import division, print_function
import numpy as np
import openmdao.api as om

class Dynamics(om.ExplicitComponent):


    def initialize(self):
        # declare the input dict provided in the run script
        self.options.declare('num_nodes', types=int)
        #self.options.declare('input_dict', types=dict)
        

    def setup(self):
        nn = self.options['num_nodes']
        #input_dict = self.options['input_dict']

        # Setup Inputs and Outputs using EQ #21 and 22
        self.add_input('T', val = 500*np.ones(nn), units='N', desc='Thrust')
        self.add_input('theta', val = 5*np.ones(nn), units='deg', desc='theta')
        self.add_input('D_fuse', val= 7*np.ones(nn), units='N', desc='Drag on the fuselage')
        self.add_input('alpha_inf', val= 0.4*np.ones(nn), units='deg', desc='frestream angle of attack')
        self.add_input('D_wings', val= 56*np.ones(nn), units='N', desc='total drag from two wings')
        self.add_input('alpha_efs', val=4*np.ones(nn), units='deg', desc='effective frestream angle of attack')
        self.add_input('L_wings', val=430*np.ones(nn), units='N', desc='total lift created by two wings')
        self.add_input('N', val=670*np.ones(nn), units='N', desc='total normal force from the propellers')
        self.add_input('m', val=833*np.ones(nn), units='kg', desc='vehicle mass')

        # Setup Outputs
        self.add_output('x_dot', val = np.ones(nn), units='m', desc='Horizontal Velocity')
        self.add_output('y_dot', val = np.ones(nn), units='m', desc='Vertical Velocity')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        # Lets be explicit so we don't miss anything
        self.declare_partials(of='x_dot', wrt=['T','theta','D_fuse', 'alpha_inf', 'D_wings', 'alpha_efs', 'L_wings','N','m'], rows=arange, cols=arange)
        self.declare_partials(of='y_dot', wrt=['T','theta','D_fuse', 'alpha_inf', 'D_wings', 'alpha_efs', 'L_wings','N','m'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        T = inputs['T']
        theta = inputs['theta']
        D_fuse = inputs['D_fuse']
        alpha_inf = inputs['alpha_inf']
        D_wings = inputs['D_wings']
        alpha_efs = inputs['alpha_efs']
        L_wings = inputs['L_wings']
        N = inputs['N']
        m = inputs['m']
        g = -9.81 # m/s gravitational attraction

        outputs['x_dot'] = ( T * np.sin(theta) - D_fuse * np.sin(theta + alpha_inf) - D_wings * np.sin(theta + alpha_efs) - L_wings * np.cos(theta + alpha_efs) - N * np.cos(theta) ) * m**(-1)
        outputs['y_dot'] = ( T * np.cos(theta) - D_fuse * np.cos(theta + alpha_inf) - D_wings * np.cos(theta + alpha_efs) + L_wings * np.sin(theta + alpha_efs) + N * np.sin(theta) -m * g) * m**(-1)

    def compute_partials(self, inputs, J):
        T = inputs['T']
        theta = inputs['theta']
        D_fuse = inputs['D_fuse']
        alpha_inf = inputs['alpha_inf']
        D_wings = inputs['D_wings']
        alpha_efs = inputs['alpha_efs']
        L_wings = inputs['L_wings']
        N = inputs['N']
        m = inputs['m']
        g = -9.81 # m/s gravitational attraction

        J['x_dot','T'] = np.sin(theta) * m**(-1)
        J['x_dot','theta'] = ( T * np.cos(theta) - D_fuse * np.cos(theta + alpha_inf) - D_wings * np.cos(theta + alpha_efs) + L_wings * np.sin(theta + alpha_efs) + N * np.sin(theta) ) * m**(-1)
        J['x_dot','D_fuse'] = - np.sin(theta + alpha_inf) * m**(-1)
        J['x_dot','alpha_inf'] = ( - D_fuse * np.cos(theta + alpha_inf)) * m**(-1)
        J['x_dot','D_wings'] = (- np.sin(theta + alpha_efs) ) * m**(-1)
        J['x_dot','alpha_efs'] = (- D_wings * np.cos(theta + alpha_efs) + L_wings * np.sin(theta + alpha_efs)) * m**(-1)
        J['x_dot','L_wings'] = (- np.cos(theta + alpha_efs)) * m**(-1)
        J['x_dot','N'] = (- np.cos(theta) ) * m**(-1)
        J['x_dot','m'] = -1 * ( T * np.sin(theta) - D_fuse * np.sin(theta + alpha_inf) - D_wings * np.sin(theta + alpha_efs) - L_wings * np.cos(theta + alpha_efs) - N * np.cos(theta) ) * m**(-2)

        J['y_dot','T'] = ( np.cos(theta)) * m**(-1)
        J['y_dot','theta'] = ( - T * np.sin(theta) + D_fuse * np.sin(theta + alpha_inf) + D_wings * np.sin(theta + alpha_efs) + L_wings * np.cos(theta + alpha_efs) + N * np.cos(theta)) * m**(-1)
        J['y_dot','D_fuse'] = (- np.cos(theta + alpha_inf)) * m**(-1)
        J['y_dot','alpha_inf'] = (D_fuse * np.sin(theta + alpha_inf)) * m**(-1)
        J['y_dot','D_wings'] = (-np.cos(theta + alpha_efs)) * m**(-1)
        J['y_dot','alpha_efs'] = (D_wings * np.sin(theta + alpha_efs) + L_wings * np.cos(theta + alpha_efs)) * m**(-1)
        J['y_dot','L_wings'] = (np.sin(theta + alpha_efs)) * m**(-1)
        J['y_dot','N'] = (np.sin(theta)) * m**(-1)
        J['y_dot','m'] = -1 * ( T * np.cos(theta) - D_fuse * np.cos(theta + alpha_inf) - D_wings * np.cos(theta + alpha_efs) + L_wings * np.sin(theta + alpha_efs) + N * np.sin(theta)) * m**(-2)

# Test partials
if __name__ == "__main__":
    from openmdao.api import Problem, Group, IndepVarComp

    p = Problem()
    p.model = Group()
    #des_vars = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    p.model.add_subsystem('Dynamics', Dynamics(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.check_partials(compact_print=True, method='cs')
