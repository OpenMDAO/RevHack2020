
import openmdao.api as om
import dymos as dm 
from dynamics_comp import Dynamics
import numpy as np 

class flight_ODE(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)  
        #self.options.declare('input_dict', type=dict)

    def setup(self):
        nn = self.options['num_nodes']
        #input_dict = self.options['input_dict']

        self.add_subsystem('Dynamics', Dynamics(num_nodes=nn), 
            promotes_inputs=['T','theta','D_fuse', 'alpha_inf', 'D_wings', 'alpha_efs', 'L_wings','N','m'],
            promotes_outputs=['x_dot','y_dot'])