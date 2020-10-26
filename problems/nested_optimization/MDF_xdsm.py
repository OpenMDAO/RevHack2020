from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC

x = XDSM()

# Systems
x.add_system("top_opt", OPT, "Optimizer")
x.add_system("sub_opt1", OPT, "Sub\_Optimizer")
x.add_system("comp_pitch_angles", FUNC, "ComputePitchAngle")
x.add_system("d_airfoil", FUNC, "DesignAirfoil")
x.add_system("comp_mod_power", FUNC, "ComputeModifiedPower")

# Connections
x.connect("comp_pitch_angles", "comp_mod_power", "powers")
x.connect("d_airfoil", "comp_mod_power", "airfoil\_design")
x.connect("comp_mod_power","top_opt", "modified\_power")
x.connect("comp_pitch_angles", "sub_opt1", "powers")
x.connect("sub_opt1", "comp_pitch_angles", "pitch\_angle")
x.connect("d_airfoil", "top_opt", "summed\_efficiency")
x.connect('top_opt', 'sub_opt1', "airfoil\_design, drag\_modifier")

# Outputs
x.add_output("sub_opt1", "airfoil\_design*, drag\_modifier*")
x.add_output("comp_mod_power", "modified\_power*")
x.add_output('top_opt', 'airfoil\_design*, drag\_modifier*')
x.add_output('comp_pitch_angles', 'total\_power*')

x.write("run_MDF")
