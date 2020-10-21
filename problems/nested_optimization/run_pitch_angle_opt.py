import numpy as np
import openmdao.api as om
from components.compute_pitch_angles import ComputePitchAngles

wind_speeds = [4.0, 6.0, 8.0, 10.0]
P_rated = 500.0

prob = om.Problem()
prob.model.add_subsystem(
    "compute_pitch_angles",
    ComputePitchAngles(size=len(wind_speeds), P_rated=P_rated),
    promotes=["*"],
)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.model.approx_totals(method="fd")

prob.model.add_design_var("drag_modifier", lower=6.0, upper=15.0)
prob.model.add_objective("total_power")

prob.setup()

prob.set_val("wind_speeds", wind_speeds)
prob.run_driver()

prob.model.list_outputs(print_arrays=True)