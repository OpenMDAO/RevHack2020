# Nested optimization makes sense for this kind of problem

Before jumping into solutions, its important to understand why sub-optimization makes sense in this kind of problem. 
Here is the general layout: 

You are designing a wind turbine, which will see a lot of different wind speeds. 
You need to make sure it performs well at all of those speeds.
So for every design, you look at a sweep of wind speeds and compute the maximum power that you can produce at each one. 
You find the maximum power by varying the pitch angle of the blades at each wind speed. 

The term *maximum power* directly implies the need for optimization, but why should it be sub-optimization? 
Overall we have two sets of design variables: 
* operational design variables: the array of pitch angles are various wind speeds
* physical design variables: the physical design of the wind turbine (e.g. tower height, diameter, blade twist, etc. )

Splitting the design variables up into these categories helps clarify the sub-optimization pictures. 
We can see that the pitch angle at each wind-speed is independent of all the other wind speeds (at least to first order), and that degree of de-coupling between design variables naturally points toward optimizing each wind speed separately. 
However, the key issue is really how strongly coupled the operational design variables are to the physical design variables. 
In other words, how strongly do changes in the turbine design effect the optimal pitch angles? 

Admittedly, no one on the OpenMDAO dev team is anything close to an expert on wind turbines ... but we'll speculate anyway :) 
The coupling physical-operational coupling here is pretty weak. 
Yes, changes to the design will absolutely affect the optimal pitch angles but they will do so in a fairly predictable and mild manner. 
In other words, the optimal pitch angles will change but not a whole lot. 
Given the weak coupling, we can reasonably guess that using sub-optimization will help to reduce the top level design space and make the top level problem easier to solve. 

# Should you use a sub-problem for nested optimization? 

There isn't a single answer to this in general. 
We can look at the specific case of the problem posed by John Jasa, involving computing the power curve for a wind turbine. 

There is one component that does the nested-optimization (i.e. no groups, connectection, etc). Also the sub-optimization in this case uses finite differences for derivatives, and in the spirit of the problem given we assumed it would stay that way. So you wouldn't get the benefit of the analytic derivatives, and there is no other features of OpenMDAO being used in the sub-optimization... hence there isn't a lot of value in it. 
Also, there is just less code without the sub-problem. 

You can see how things line up against it in this case. If you wanted to switch to analtyic derivatives, or if your sub-problem involved more anlyses that were coupled together then you'd probably be better off with a sub-problem. 

## Code with out a sub-problem: 

```python 
import numpy as np
from scipy.optimize import minimize
import openmdao.api as om


def compute_power(pitch_angle, wind_speed, drag_modifier):
    CD = np.pi * drag_modifier * np.deg2rad(pitch_angle) ** 2
    airfoil_power_boost = (drag_modifier - wind_speed * 2.0) ** 2.0 / 10.0
    return -((wind_speed - CD) ** 3) - airfoil_power_boost


def compute_power_constraint(pitch_angle, wind_speed, drag_modifier, P_rated):
    neg_power = compute_power(pitch_angle, wind_speed, drag_modifier)
    return neg_power + P_rated


class ComputePitchAngles(om.ExplicitComponent):
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

        for i, wind_speed in enumerate(inputs["wind_speeds"]):
            constraints = [
                {
                    "type": "ineq",
                    "fun": compute_power_constraint,
                    "args": [wind_speed, drag_modifier, P_rated],
                }
            ]
            result = minimize(
                compute_power,
                1.0,
                args=(wind_speed, drag_modifier),
                method="SLSQP",
                bounds=[(-15.0, 15.0)],
                options={"disp": False},
                constraints=constraints,
            )
            outputs["pitch_angles"][i] = result.x
            outputs["powers"][i] = result.fun

        outputs["total_power"] = np.sum(outputs["powers"])
```

## Code with a sub-problem 

```python 
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

        self._problem = prob = om.Problem()
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

    def compute(self, inputs, outputs):
        P_rated = self.options["P_rated"]
        drag_modifier = inputs["drag_modifier"]

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
```


# Our solutions: 
## With sub-problems (just for demonstration purposes)
* [Nested optimization with a sub-problem](./run_sequential_opt_using_subproblem.py)
* [Sequential Optimization with sub-problem](./run_sequential_opt_using_subproblem.py)

## Solutions without sub-problems
* Original solution given by John Jasa
* Solver based approach


# A theoretical discussion on nested optimizations

## You can think of nested optimization as a form of Multi Disciplinary Feasbile (MDF) architecture

The MDF architecture is typically thought of in terms of converging the governing equations for your analysis with a solver. 
Thats an overly specific interpretation though. 
More generally, MDF removes degrees of freedom from the top level problem by handing them to a well behaved sub-solver. 

If that sub-solver happens to be a nonlinear solver and the degrees of freedom happen to be state variables for your analysis then you get the traditional view of MDF. 
But its equally valid to think of the sub-solver as an optimizer and the degrees of freedom as the operational variables for your problem. 

Generally speaking we know that the MDF solution approach tends to be slower but more reliably convergent, assuming you can reliably get a converged sub-problem for any design within the search space of the top level problem. 

So in the context of this kind of problem, it seem to us that using an MDF style approach is reasonable. 

## What about seqential optimization? 

This approach is very commonly used. 
You break a problem into two parts, optimize each one separately and then iterate between them till things damp out. 

Is it a good idea? It is not possible to give a definitive answer. Sometimes yes, sometimes no. 
It is highly problem specific. 
If you want our opinion, we don't like this approach because there are common situations where it misses important couplings and gives you a poor answer. 
That won't always happen, but it happens often enough that we don't like this approach. 

Others might say "I know it doesn't give the best answer but it gives a better one than my baseline and that is good enough!" 
That is a totally valid opinion. 
Its not critical that you *always* get the best possible optimum, and if sequential optimization is working for your application then its a nice way to simplify the problem. 

Just keep in mind that you may be leaving performance on the table, 
and if you are in a situation where you're not happy with the answer from your sequential opt then its worth considering a more fully coupled solution approach. 

# Nested Optimization Solutions

If you want to use sub-optimization, you take one of two approaches: 

* Wrap up a chunk of your model into a sub-problem and then embed that into a component in a larger model 
* Write your own custom optimization routine into the `compute` method of a component. 

We want to stress that BOTH of these approaches are reasonable! 


# Solvers: Alternative to Nested Optimization
