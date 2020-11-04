import numpy as np
from scipy.optimize import brentq
import openmdao.api as om


def compute_power(pitch_angle, wind_speed, drag_modifier):
    CD = np.pi * drag_modifier * np.deg2rad(pitch_angle) ** 2
    airfoil_power_boost = (drag_modifier - wind_speed * 2.0) ** 2.0 / 10.0
    return -((wind_speed - CD) ** 3) - airfoil_power_boost


def fd_dpower__dpitch_angle(pitch_angle, wind_speed, drag_modifier): 
    '''central difference approximation of dpower__dpitch_angle'''

    step = 1e-4
    p = compute_power(pitch_angle, wind_speed, drag_modifier)
    p_minus = compute_power(pitch_angle-step, wind_speed, drag_modifier)
    p_plus = compute_power(pitch_angle+step, wind_speed, drag_modifier)

    return p, (p_plus - p_minus)/(2*step)

def composite_residual(pitch_angle, wind_speed, drag_modifier, P_rated, return_power=False): 
    ''' a "trick" to apply a constraint is to conditionally evaluate different residuals''' 

    # NOTES: This "trick" works fine as long as one of two conditions is met: 
    # 1) Your residuals are c1 continuous across the conditional 
    # 2) Your residuals are c0 continuous and  you don't end up oscillating 
    #     back and forth across the breakpoint in the conditional

    power, d_power = fd_dpower__dpitch_angle(pitch_angle, wind_speed, drag_modifier)


    if power < -P_rated: 
        R = power-P_rated
    else: 
        R = d_power

    if return_power: 
        return R, power
    else: 
        return R


# If you are going to globally finite difference over an entire model, which this is a part of
# then the explicit component works just as well as the implicit one
class ComputePitchAnglesSolverExplicit(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("size")
        self.options.declare("P_rated")

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

        RETURN_POWER = False

        for i, wind_speed in enumerate(inputs["wind_speeds"]):
            root = brentq(composite_residual, -15, 15, 
                          args=(wind_speed, drag_modifier, P_rated, RETURN_POWER))
            outputs["pitch_angles"][i] = root
            outputs["powers"][i] = compute_power(root, wind_speed, drag_modifier)

        outputs["total_power"] = np.sum(outputs["powers"])


# If you are going to be finite differencing the partial derivatives of the component 
# then an implicit component will be more efficient and numerically stable
class ComputePitchAnglesSolverImplicit(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("size")
        self.options.declare("P_rated")

    def setup(self):
        size = self.options["size"]

        self.add_input("wind_speeds", np.zeros(size))
        self.add_input("drag_modifier", 11.0)

        self.add_output("pitch_angles", np.zeros(size))
        self.add_output("powers", np.zeros(size))
        self.add_output("total_power")

    def apply_nonlinear(self, inputs, outputs, residuals): 
        P_rated = self.options["P_rated"]
        drag_modifier = inputs["drag_modifier"]

        RETURN_POWER = True

        for i, wind_speed in enumerate(inputs["wind_speeds"]):
            pitch_angle = outputs['pitch_angles'][i]
            R, power = composite_residual(pitch_angle, wind_speed, drag_modifier, RETURN_POWER)
            residuals['pitch_angles'][i] = R
            residauls['powers'][i] = outputs['powers'][i] - power

    def solve_nonlinear(self, inputs, outputs):
        P_rated = self.options["P_rated"]
        drag_modifier = inputs["drag_modifier"]

        RETURN_POWER = False

        for i, wind_speed in enumerate(inputs["wind_speeds"]):
            root = brentq(composite_residual, -15, 15, 
                          args=(wind_speed, drag_modifier, P_rated, RETURN_POWER))
            outputs["pitch_angles"][i] = root
            outputs["powers"][i] = compute_power(root, wind_speed, drag_modifier)

        outputs["total_power"] = np.sum(outputs["powers"])

if __name__ == "__main__": 

    from compute_pitch_angles import ComputePitchAngles
    wind_speeds = [4.0, 6.0, 8.0, 10.0]
    P_rated = 500.0

    p = om.Problem()
    p.model = ComputePitchAngles(P_rated=P_rated, size=4)
    p.setup()
    p['wind_speeds'] = wind_speeds
    p.run_model()
    expected = p['powers'].copy()

    p = om.Problem()
    p.model = ComputePitchAnglesSolverExplicit(P_rated=P_rated, size=4)
    p.setup()
    p['wind_speeds'] = wind_speeds
    p.run_model()
    computed_explicit = p['powers'].copy()

    p = om.Problem()
    p.model = ComputePitchAnglesSolverImplicit(P_rated=P_rated, size=4)
    p.setup()
    p['wind_speeds'] = wind_speeds
    p.run_model()
    computed_implicit = p['powers'].copy()

    print('expected', expected)
    print('computed explicit', computed_explicit)
    print('computed implicit', computed_implicit)

