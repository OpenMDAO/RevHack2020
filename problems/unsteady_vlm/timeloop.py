from __future__ import division
from openmdao.api import Component, Group

from transfer import TransferDisplacements, TransferLoads
from uvlm import UVLMStates
from spatialbeam import SpatialBeamStates


class SingleStep(Group):
    """ Group that contains components that have to be run for each time step """

    def __init__(self, num_x, num_y, num_w, E, G, mrho, fem_origin, SBEIG, t):
        super(SingleStep, self).__init__()

        name_def_mesh = 'def_mesh_%d'%t
        name_vlmstates = 'vlmstates_%d'%t
        name_loads = 'loads_%d'%t
        name_spatialbeamstates = 'spatialbeamstates_%d'%t

        self.add(name_def_mesh,
                 TransferDisplacements(num_x, num_y, t, fem_origin),
                 promotes=['*'])
        self.add(name_vlmstates,
                 UVLMStates(num_x, num_y, num_w, t),
                 promotes=['*'])
        self.add(name_loads,
                 TransferLoads(num_x, num_y, t, fem_origin),
                 promotes=['*'])
        self.add(name_spatialbeamstates,
                 SpatialBeamStates(num_x, num_y, E, G, mrho, SBEIG, t),
                 promotes=['*'])

class SingleAeroStep(Group):
    """ Group that contains components that have to be run for each time step """

    def __init__(self, num_x, num_y, num_w, E, G, mrho, fem_origin, t):
        super(SingleAeroStep, self).__init__()

        name_def_mesh = 'def_mesh_%d'%t
        name_vlmstates = 'vlmstates_%d'%t
        name_loads = 'loads_%d'%t
        name_spatialbeamstates = 'spatialbeamstates_%d'%t

        self.add(name_def_mesh,
              TransferDisplacements(num_x, num_y, t, fem_origin),
              promotes=['*'])
        self.add(name_vlmstates,
              UVLMStates(num_x, num_y, num_w, t),
              promotes=['*'])
        self.add(name_loads,
              TransferLoads(num_x, num_y, t, fem_origin),
              promotes=['*'])

class SingleStructStep(Group):
    """ Group that contains components that have to be run for each time step """

    def __init__(self, num_x, num_y, num_w, E, G, mrho, fem_origin, SBEIG, t):
        super(SingleStructStep, self).__init__()

        name_def_mesh = 'def_mesh_%d'%t
        name_loads = 'loads_%d'%t
        name_spatialbeamstates = 'spatialbeamstates_%d'%t

        # self.add(name_def_mesh,
        #          TransferDisplacements(num_x, num_y, t, fem_origin),
        #          promotes=['*'])
        # self.add(name_loads,
        #          TransferLoads(num_x, num_y, t, fem_origin),
        #          promotes=['*'])
        self.add(name_spatialbeamstates,
                 SpatialBeamStates(num_x, num_y, E, G, mrho, SBEIG, t),
                 promotes=['*'])
