import numpy as np
import openmdao.api as om


def compute_power(pitch_angle, wind_speed, drag_modifier):
    CD = np.pi * drag_modifier * np.deg2rad(pitch_angle) ** 2
    airfoil_power_boost = (drag_modifier - wind_speed * 2.0) ** 2.0 / 10.0
    return -((wind_speed - CD) ** 3) - airfoil_power_boost

def compute_power_constraint(pitch_angle, wind_speed, drag_modifier, P_rated):
    neg_power = compute_power(pitch_angle, wind_speed, drag_modifier)
    return neg_power + P_rated

class ComputePower(om.ExplicitComponent):
    def setup(self):
        self.add_input("pitch_angle", 0.0)
        self.add_input("wind_speed", 0.0)
        self.add_input("drag_modifier", 0.0)
        self.add_input("P_rated", 0.0)

        self.add_output("power")
        self.add_output("power_constraint")

    def compute(self, inputs, outputs):
        outputs["power"] = compute_power(
            inputs["pitch_angle"],
            inputs["wind_speed"],
            inputs["drag_modifier"])

        outputs["power_constraint"] = compute_power_constraint(
            inputs["pitch_angle"],
            inputs["wind_speed"],
            inputs["drag_modifier"],
            inputs["P_rated"])

class ComputePitchAnglesUsingSubProblem(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("size")
        self.options.declare("P_rated")
        self._problem = None

    def setup(self):
        size = self.options["size"]

        self.add_input("wind_speeds", np.zeros(size))
        self.add_input("drag_modifier", 11.0)

        self.add_output("pitch_angles", np.zeros(size))
        self.add_output("powers", np.zeros(size))
        self.add_output("total_power")

    def compute(self, inputs, outputs):
        P_rated = self.options["P_rated"]
        drag_modifier = inputs["drag_modifier"]

        if self._problem is None:
            prob = om.Problem()
            self._problem = prob
            prob.model.add_subsystem(
                "compute_power",
                ComputePower(),
                promotes=["*"],
            )

            prob.driver = om.ScipyOptimizeDriver()
            prob.driver.options["optimizer"] = "COBYLA"
            prob.model.approx_totals(method="fd")

            prob.model.add_design_var("pitch_angle", lower=-15.0, upper=15.0)
            prob.model.add_constraint("power_constraint", lower=0.0)
            prob.model.add_objective("power")

            prob.setup()

        prob = self._problem
        prob.set_val("drag_modifier", drag_modifier)
        prob.set_val("P_rated", P_rated)

        for i, wind_speed in enumerate(inputs["wind_speeds"]):
            prob.set_val("wind_speed", wind_speed)
            print("inputs before run", prob["pitch_angle"], prob["wind_speed"], prob["drag_modifier"], prob["P_rated"])
            prob.run_driver()
            outputs["pitch_angles"][i] = prob["pitch_angle"]
            outputs["powers"][i] = prob["power"]

            prob.model.list_inputs()
            prob.model.list_outputs()

        outputs["total_power"] = np.sum(outputs["powers"])

