from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, GROUP

x = XDSM()

x.add_system("prob1_opt", OPT, "Optimize\_Prob1")
x.add_system("d_airfoil", FUNC, "DesignAirfoil", stack=True)
x.add_system("comp_mod_power", FUNC, "ComputeModifiedPower")

x.connect("prob1_opt", "d_airfoil", "airfoil\_design")
x.connect("comp_mod_power","prob1_opt", "modified\_power")
x.connect("d_airfoil","comp_mod_power", "aerodynamic\_eff", stack=True)
x.connect("d_airfoil","comp_mod_power2", "aerodynamic\_eff*", stack=True)

x.add_input("d_airfoil", "wind\_speeds", stack=True)

x.add_system("prob2_opt", OPT, "Optimize\_Prob2")
x.add_system("comp_pitch_angles", GROUP, "ComputeOptPitchAngle", stack=True)
x.add_system("comp_mod_power2", FUNC, "ComputeModifiedPower")

x.connect("prob2_opt", "comp_pitch_angles", "drag\_modifier")
x.connect("comp_mod_power2","prob2_opt", "powers, modified\_power")
x.connect("comp_pitch_angles", "comp_mod_power2", "powers*", stack=True)
x.connect("comp_pitch_angles", "comp_mod_power", "powers**", stack=True)

x.add_input("comp_pitch_angles", "wind\_speeds", stack=True)

x.write("run_sequential_top_level")