import openmdao.api as om

from transfer import TransferDisplacements, TransferLoads
from uvlm import UVLMStates
from spacialbeam import SpatialBeamStates


class SingleStep(om.Group):
    """ Group that contains components that have to be run for each time step """

    def __init__(self, num_x, num_y, num_w, E, G, mrho, fem_origin, SBEIG, t):
        super(SingleStep, self).__init__()

        name_def_mesh = 'def_mesh_%d'%t
        name_vlmstates = 'vlmstates_%d'%t
        name_loads = 'loads_%d'%t
        name_spatialbeamstates = 'spatialbeamstates_%d'%t

        self.add_subsystem(name_def_mesh,
                 TransferDisplacements(nx=num_x, n=num_y, t=t, fem_origin=fem_origin),
                 promotes=['*'])
        self.add_subsystem(name_vlmstates,
                 UVLMStates(num_x, num_y, num_w, t),
                 promotes=['*'])
        self.add_subsystem(name_loads,
                 TransferLoads(nx=num_x, n=num_y, t=t, fem_origin=fem_origin),
                 promotes=['*'])
        self.add_subsystem(name_spatialbeamstates,
                 SpatialBeamStates(num_x, num_y, E, G, mrho, SBEIG, t),
                 promotes=['*'])



class TimeLoopComp(om.ExplicitComponent): 

  def __init__(self, num_x, num_y, num_w, E, G, mrho, fem_origin, SBEIG, num_times):

        self.num_x = num_x 
        self.num_y = num_y 
        self.num_w = num_w
        self.E = E 
        self.G = G 
        self.mrho = mrho 
        self.fem_origin = fem_origin
        self.SBEIG = SBEIG
        self.num_times = num_times 

        super().__init__()



  def setup(self): 

      self.add_input('dt')
      self.add_input('K_matrix', shape=(self.num_y*6,self.num_y*6))
      self.add_input('mesh', shape=(self.num_x, self.num_y, 3))
      self.add_input('loads', val=0.)
      self.add_input('rho')
      self.add_input('alpha')
      self.add_input('v')

      self.add_output('S_ref')
      self.add_output('sec_L_19', shape=(self.num_x-1, self.num_y-1))
      self.add_output('sec_D_19', shape=(self.num_x-1, self.num_y-1))


      p0 = self._prob0 = om.Problem()

      # NOTE: In the original code `t` was used in the variable naming scheme
      #       and there were some slight differences between I/O from t=0 and t>0
      #       so we need to have both
      p0.model = SingleStep(self.num_x, self.num_y, self.num_w, self.E, 
                           self.G, self.mrho, self.fem_origin, self.SBEIG, t=0)

      p0.setup()
      p0.final_setup() 


     
  def compute(self, inputs, outputs): 

    num_dt = self.num_times 
    p0 = self._prob0

    p0['dt'] = inputs['dt']
    p0['K_matrix'] = inputs['K_matrix']
    p0['mesh'] = inputs['mesh']
    p0['loads'] = inputs['loads']
    p0['rho'] = inputs['rho']
    p0['alpha'] = inputs['alpha']
    p0['v'] = inputs['v']

    p0.run_model()

    circ = p0['circ_0'].copy()
    # circ_wake = p0['circ_wake_0'].copy()
    wake_mesh = p0['wake_mesh_1'].copy()
    sigma_x = p0['sigma_x_0'].copy()
    disp = p0['disp_0']

    for t in range(1,num_dt):

        pi = om.Problem()

        # NOTE: The size of the wake mesh grows with each iteration, so we need to re-do setup each time
        pi.model.add_subsystem('step', SingleStep(self.num_x, self.num_y, self.num_w, self.E, 
                               self.G, self.mrho, self.fem_origin, self.SBEIG, t=t), 
                               promotes=['*'])

        pi.setup()
        pi.final_setup()

        pi['dt'] = inputs['dt']
        pi['K_matrix'] = inputs['K_matrix']
        pi['mesh'] = inputs['mesh']
        pi['loads'] = inputs['loads']
        pi['rho'] = inputs['rho']
        pi['alpha'] = inputs['alpha']
        pi['v'] = inputs['v']

        pi[f'circ_{t-1}'] = circ
        if t > 1: 
          pi[f'circ_wake_{t-1}'] = circ_wake
        pi[f'wake_mesh_{t}'] = wake_mesh
        pi[f'sigma_x_{t-1}'] = sigma_x
        pi[f'disp_{t-1}'] = disp

        pi.run_model()

        # save the data to pass to the next time instance
        circ = pi[f'circ_{t}'].copy()
        circ_wake = pi[f'circ_wake_{t}'].copy()
        wake_mesh = pi[f'wake_mesh_{t+1}'].copy()
        sigma_x = pi[f'sigma_x_{t}'].copy()
        disp = pi[f'disp_{t}'].copy()

    outputs['S_ref'] = p0['S_ref']
    outputs['sec_L_19'] = pi['sec_L_19']
    outputs['sec_D_19'] = pi['sec_D_19']
