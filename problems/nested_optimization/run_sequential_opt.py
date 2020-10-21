import numpy as np
import openmdao.api as om
from components.compute_pitch_angles import ComputePitchAngles
from components.design_airfoil import DesignAirfoil
from components.compute_modified_power import ComputeModifiedPower


wind_speeds = [4.0, 6.0, 8.0, 10.0]
P_rated = 500.0

prob1 = om.Problem()
prob1.model.add_subsystem(
    "design_airfoil",
    DesignAirfoil(size=len(wind_speeds)),
    promotes=["*"],
)
prob1.model.add_subsystem(
    "compute_modified_power",
    ComputeModifiedPower(size=len(wind_speeds)),
    promotes=["*"],
)

prob1.driver = om.ScipyOptimizeDriver()
prob1.driver.options["optimizer"] = "SLSQP"
prob1.model.approx_totals(method="fd")

prob1.model.add_design_var("airfoil_design", lower=6.0, upper=15.0)
prob1.model.add_objective("modified_power")

prob1.setup()

prob1.set_val("wind_speeds", wind_speeds)


prob2 = om.Problem()
prob2.model.add_subsystem(
    "compute_pitch_angles",
    ComputePitchAngles(size=len(wind_speeds), P_rated=P_rated),
    promotes=["*"],
)
prob2.model.add_subsystem(
    "compute_modified_power",
    ComputeModifiedPower(size=len(wind_speeds)),
    promotes=["*"],
)

prob2.driver = om.ScipyOptimizeDriver()
prob2.driver.options["optimizer"] = "SLSQP"
prob2.model.approx_totals(method="fd")

prob2.model.add_design_var("drag_modifier", lower=6.0, upper=15.0)
prob2.model.add_constraint("powers", lower=-P_rated)
prob2.model.add_objective("modified_power")

prob2.setup()

prob2.set_val("wind_speeds", wind_speeds)



for i in range(10):
    prob1.set_val("powers", prob2["powers"])
    prob1.run_driver()
    prob1.model.list_inputs()
    prob1.model.list_outputs(print_arrays=True)

    prob2.set_val("aerodynamic_efficiency", prob1["aerodynamic_efficiency"])
    prob2.run_driver()
    prob2.model.list_inputs()
    prob2.model.list_outputs(print_arrays=True)