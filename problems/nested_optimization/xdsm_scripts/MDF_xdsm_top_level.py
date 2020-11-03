from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, GROUP

x = XDSM()

# Systems
x.add_system("top_opt", OPT, "Optimizer")
x.add_system("comp_pitch_angles", GROUP, "ComputeOptPitchAngle", stack=True)
x.add_system("d_airfoil", FUNC, "DesignAirfoil", stack=True)
x.add_system("comp_mod_power", FUNC, "ComputeModifiedPower")

x.connect("d_airfoil", "comp_mod_power", "aerodynamic\_eff", stack=True)
x.connect("comp_pitch_angles", "comp_mod_power", "powers*", stack=True)
x.connect("top_opt", "d_airfoil", "airfoil\_design")
x.connect("top_opt", "comp_pitch_angles", "drag\_modifier")
x.connect("comp_mod_power", "top_opt", "modified\_power")

x.add_input("d_airfoil", "wind\_speeds", stack=True)
x.add_input("comp_pitch_angles", "wind\_speeds", stack=True)

x.write("run_MDF_top_level")