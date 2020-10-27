from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, GROUP

x = XDSM()

x.add_system("top_opt", OPT, "Optimizer")
x.add_system("sub_opt", OPT, "Sub\_Optimizer")
x.add_system("comp_pitch_angles", FUNC, "ComputePitchAngle", stack=True)

x.connect('top_opt', 'comp_pitch_angles', 'drag\_modifier')
x.connect('comp_pitch_angles', 'top_opt', 'total\_power')
x.connect("sub_opt", "comp_pitch_angles", "pitch\_angle", stack=True)
x.connect("comp_pitch_angles", "sub_opt", "powers", stack=True)

x.add_input("comp_pitch_angles", "wind\_speeds", stack=True)

x.write("run_comp_pitch_angles_sub_opt")