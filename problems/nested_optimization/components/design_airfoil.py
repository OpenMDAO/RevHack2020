import numpy as np
from scipy.optimize import minimize
import openmdao.api as om


class DesignAirfoil(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("size")

    def setup(self):
        size = self.options["size"]

        self.add_input("wind_speeds", np.zeros(size))
        self.add_input("airfoil_design", 11.0)

        self.add_output("aerodynamic_efficiency", np.zeros(size))
        self.add_output("summed_efficiency")

    def compute(self, inputs, outputs):
        airfoil_design = inputs["airfoil_design"]

        for i, wind_speed in enumerate(inputs["wind_speeds"]):
            lift = (airfoil_design - 7) * wind_speed
            drag = (airfoil_design * wind_speed) ** 2
            outputs["aerodynamic_efficiency"][i] = lift / drag

        outputs["summed_efficiency"] = -np.sum(outputs["aerodynamic_efficiency"])


if __name__ == "__main__":
    wind_speeds = [4.0, 6.0, 8.0, 10.0]

    prob = om.Problem()
    prob.model.add_subsystem(
        "design_airfoil",
        DesignAirfoil(size=len(wind_speeds)),
        promotes=["*"],
    )

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.model.approx_totals(method="fd")

    prob.model.add_design_var("airfoil_design", lower=6.0, upper=15.0)
    prob.model.add_objective("summed_efficiency")

    prob.setup()

    prob.set_val("wind_speeds", wind_speeds)
    prob.run_driver()

    prob.model.list_inputs()
    prob.model.list_outputs(print_arrays=True)
