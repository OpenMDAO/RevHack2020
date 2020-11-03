from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, GROUP

x = XDSM()

x.add_system("top_opt", OPT, "Optimizer")
x.add_system("comp_pitch_angles", GROUP, "ComputeOptPitchAngle", stack=True)

x.connect('top_opt', 'comp_pitch_angles', 'drag\_modifier')
x.connect('comp_pitch_angles', 'top_opt', 'total\_power')

x.add_input("comp_pitch_angles", "wind\_speeds", stack=True)

x.write("run_comp_pitch_angles")