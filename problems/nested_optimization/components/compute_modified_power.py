import numpy as np
from scipy.optimize import minimize
import openmdao.api as om


class ComputeModifiedPower(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("size")

    def setup(self):
        size = self.options["size"]

        self.add_input("aerodynamic_efficiency", np.zeros(size))
        self.add_input("powers", np.zeros(size))

        self.add_output("modified_power")

    def compute(self, inputs, outputs):
        outputs["modified_power"] = np.sum((1. + inputs["aerodynamic_efficiency"])**3 * inputs["powers"])