""" Code to run aerostructural analysis and evaluate flutter velocity.
Call as `python run_aerostruct.py` to run a single analysis. """

from __future__ import division
import numpy
from time import time

from openmdao.api import IndepVarComp, Problem, Group, ScipyOptimizer, Newton, ScipyGMRES, LinearGaussSeidel, NLGaussSeidel, SqliteRecorder, profile
from geometry import GeometryMesh, gen_crm_mesh, gen_mesh
from materials import MaterialsTube
from spatialbeam import SpatialBeamMatrices, SpatialBeamEIG, radii
from timeloop import SingleStep
from uvlm import UVLMFunctionals
from openmdao.devtools.partition_tree_n2 import view_tree
import warnings
warnings.filterwarnings("ignore")

#######################################
# Define loop for parametric analysis #
#######################################
num_of_points = 1
num_of_angles = 1
velocities_vect = numpy.linspace(10.0, 50.0, num=num_of_points)
#num_dt_vect = numpy.linspace(100, 400, num=num_of_points)
#zeta_vect = numpy.linspace(0., 1., num=num_of_points)
alpha_vect = numpy.linspace(2.0, 2.0, num=num_of_angles)

for p in xrange(num_of_angles):
    for i in xrange(num_of_points):

        v = velocities_vect[i]
        #v = 200.                       # flow speed [m/s]
        v = float(v)

        ############################################
        # Define parameters for simulation in time #
        ############################################
        #num_dt = int(num_dt_vect[i])
        num_dt = 20                     # number of time steps
        final_t = 1.                    # time-simulation duration [s]
        num_w = 20                      # number of (timewise) deforming wake elements

        #####################################
        # Define the aerodynamic parameters #
        #####################################
        rho = 0.0889                    # air density [kg/m^3]
        alpha = float(alpha_vect[p])
        #alpha = 0.5                        # angle of attack [deg.]
        CL0 = 0.
        CD0 = 0.

        #############################################
        # Define wing geometry and aerodynamic mesh #
        #############################################
        CRM = 0

        if CRM:                         # Use the CRM wing model
            wing = 'CRM'
            npi = 2                 # number of points inboard
            npo = 3                     # number of points outboard
            full_wing_mesh = gen_crm_mesh(npi, npo, num_x=5)
            num_x, num_y = full_wing_mesh.shape[:2]
            num_y_sym = numpy.int((num_y + 1) / 2)
            span = 58.7630524 # [m]
            num_twist = 5

        else:                           # Use the rectangular wing model
            wing = 'RECT'
            num_x = 3                   # number of spanwise nodes
            num_y = 11                  # number of chordwise nodes
            num_y_sym = numpy.int((num_y + 1) / 2)
            span = 32. # [m]
            chord = 1. # [m]
            cosine_spacing = 0.
            full_wing_mesh = gen_mesh(num_x, num_y, span, chord, cosine_spacing)
            num_twist = numpy.max([int((num_y - 1) / 5), 5])

        half_wing_mesh = full_wing_mesh[:, (num_y_sym-1):, :]

        ##########################
        # Define beam properties #
        ##########################
        r = radii(half_wing_mesh)/5     # beam radius
        thick = r / 5                   # beam thickness
        fem_origin = 0.5                # elastic axis position along the chord
        #zeta = zeta_vect[i]
        zeta = 0.0                      # damping percentual coeff.
        zeta = float(zeta)

        E = 70.e9 # [Pa]
        poisson = 0.3
        G = E / (2 * (1 + poisson))
        mrho = 2800. # [kg/m^3]


        ####################################
        # Define the independent variables #
        ####################################
        indep_vars = [
            ('span', span),
            ('twist', numpy.zeros(num_twist)),
            ('dihedral', 0.),
            ('sweep', 0.),
            ('taper', 1.0),
            ('v', v),
            ('alpha', alpha),
            ('rho', rho),
            ('r', r),
            ('thick', thick),
            ('zeta', zeta)]

        #######################
        # Calls of components #
        #######################
        root = Group()

        # Components before the time loop
        root.add('indep_vars',
                 IndepVarComp(indep_vars),
                 promotes=['*'])
        root.add('tube',
                 MaterialsTube(num_y_sym),
                 promotes=['*'])
        root.add('mesh',
                 GeometryMesh(full_wing_mesh, num_twist),
                 promotes=['*'])
        root.add('matrices',
                 SpatialBeamMatrices(num_x, num_y_sym, E, G, mrho, fem_origin),
                 promotes=['*'])
        SBEIG = SpatialBeamEIG(num_y_sym, num_dt, final_t)
        root.add('eig',
                 SBEIG,
                 promotes=['*'])

        # Time loop
        coupled = Group()
        for t in xrange(num_dt):
            name_step = 'step_%d'%t
            coupled.add(name_step,
                        SingleStep(num_x, num_y_sym, num_w, E, G, mrho, fem_origin, SBEIG, t),
                        promotes=['*'])

        # Set solver properties for the coupled group
        coupled.ln_solver = ScipyGMRES()
        coupled.ln_solver.options['iprint'] = 1
        coupled.ln_solver.preconditioner = LinearGaussSeidel()

        coupled.nl_solver = NLGaussSeidel()
        coupled.nl_solver.options['iprint'] = 1

        root.add('coupled',
                 coupled,
                 promotes=['*'])

        # Components after the time loop
        root.add('vlm_funcs',
                 UVLMFunctionals(num_x, num_y_sym, CL0, CD0, num_dt),
                 promotes=['*'])

        ###################
        # Run the program #
        ###################

        prob = Problem()
        prob.root = root
        prob.print_all_convergence()

        # Setup data recording
        # name_data = '%s_v%.2f_ndt%.0f_damp%.2f_alpha%.1f_w%.0f'%(wing, v, num_dt, zeta, alpha, num_w)
        # db_name = 'results/flutter/db/%s'%(name_data)
        # prob.driver.add_recorder(SqliteRecorder(db_name))

        prob.setup()
        view_tree(prob, outfile="aerostruct.html", show_browser=False)
        st = time()
        prob.run_once()

        print
        print "run time", time() - st
        print
        print "number of steps =", num_dt
        print "dt =", prob['dt']
        print "CL =", prob['CL'], "; CD =", prob['CD']
