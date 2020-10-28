# Nested Optimizations

A fundamental part of OpenMDAO is the optimizers which allow user models to find the optimal answer for their model. OpenMDAO has provided an array of different optimizers for users to choose from. As users have built more and more complex models, the need for a more clear detailing of best practices for nested optimizers has increased. The question from John Jasa highlighted this in his [submission](https://github.com/OpenMDAO/RevHack2020/tree/master/problems/nested_optimization) as his organization uses OpenMDAO and WISDEM which has nested optimization. Below we look to answer that question through examples of how to use sub problems with OpenMDAO to enhance readability and increase flexibility. In addition, highlight a problem with SLSQP optimizers when nested.

### About the Problem

We’d like to acknowledge John Jasa for the test secripts for us to work with. To demonstrate our suggestions, tricks, and things to watch out for, we will use his run_MDF_opt.py script. To provide a brief overview of the script, we are given three files. Design_airfoil.py has an ExplicitComponent which describes the lift and drag of an airfoil and outputs the summed efficiency. Compute_modified_power.py computes the modified power with respect to the given power and aerodynamic_efficiency. Finally, compute_pitch_angles.py takes the wind_speeds and drag modifier as inputs and outputs the total power. 

### Using Sub Problems in OpenMDAO 

One of the proposed problems for this reverse hackathon involved nested optimizations as described [here](https://github.com/OpenMDAO/RevHack2020/blob/master/problems/nested_optimization/readme.md)

The user requested guidance on best practices for handling these kinds of problems in OpenMDAO. This document and the code it references attempts to answer this question.

In this code, an OpenMDAO Problem and optimizer are used at the top level but for one of the Components in the model, which did some optimization, scipy’s minimize function was used. 

Although it is not well-known, OpenMDAO does support sub Problems and can be used for these situations. See [here](https://github.com/OpenMDAO/RevHack2020/tree/master/solution_approaches/sub_problem) for more discussion about OpenMDAO’s support for sub Problems.

The user provided original implementation used the following code to do the inner optimization inside the component:

```python
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

The OpenMDAO equivalent is given here. Notice that a new ExplicitComponent that can be added to the model of the Problem. The rest of the code is standard OpenMDAO code for making a model, adding and optimizers, and adding design variables, objectives, and constraints, and finally running the driver. The driver used is ScipyOptimizeDriver so that we can also use SLSQP as in the original code.

```python
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

prob = om.Problem()
prob.model.add_subsystem(
   "compute_power",
   ComputePower(),
   promotes=["*"],
)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.model.approx_totals(method="fd")

prob.model.add_design_var("pitch_angle", lower=-15.0, upper=15.0)
prob.model.add_constraint("power_constraint", lower=0.0)
prob.model.add_objective("power")

prob.setup()

prob.set_val("P_rated", P_rated)
prob.set_val("drag_modifier", drag_modifier)
prob.set_val("wind_speed", wind_speed)

prob.run_driver()

outputs["pitch_angles"][i] = prob["pitch_angle"]
outputs["powers"][i] = prob["power"]

```

### Create the Sub Problem Only Once

A way to improve the efficiency of this is to only create the sub Problem once. Then only the inputs need to be set and run_driver called. How much time this will save depends on your problem. Now the code for the component to compute the pitch angles looks like this, with the key changes noted.

```python
class ComputePitchAnglesUsingSubProblem(om.ExplicitComponent):

   def initialize(self):
       self.options.declare("size")
       self.options.declare("P_rated")
       # Added self._problem
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

       # Added check if self._problem is none
       if self._problem is None:
           prob = om.Problem()
           # Set self._problem here
           self._problem = prob
           prob.model.add_subsystem(
               "compute_power",
               ComputePower(),
               promotes=["*"],
           )

           prob.driver = om.ScipyOptimizeDriver()
           prob.driver.options["optimizer"] = "SLSQP"
           # prob.driver.options["optimizer"] = "COBYLA"
           prob.model.approx_totals(method="fd")

           prob.model.add_design_var("pitch_angle", lower=-15.0, upper=15.0)
           prob.model.add_constraint("power_constraint", lower=0.0)
           prob.model.add_objective("power")

           prob.setup()

       # Set prob here 
       prob = self._problem
       prob.set_val("drag_modifier", drag_modifier)
       prob.set_val("P_rated", P_rated)

       ## Problem sub
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



### Optimizers Compatible with Nested Optimization

One consideration with nested optimizations is that some optimizers are not re-entrant. Re-entrant is a computer science term that refers to the ability to call a function while a previous call to that function has not been completed. This is the case when doing nested optimizations. If the optimizers were not written with this in mind, the results will be incorrect.
In particular, it is known that the current implementation of SLSQP used in scipy’s minimize function is not re-entrant. Therefore, do not use SLSQP at multiple levels when using sub problems. ( Fortunately, there are implementations of SLSQP that are re-entrant, but unfortunately, they have not made their way into scipy. Here is an implementation from [Jacob Williams](https://github.com/jacobwilliams/slsqp)).
With OpenMDAO’s ScipyOptimizeDriver, one optimizer that is re-entrant is COBYLA. So the user could use SLSQP at one level and COBLYA at the other. Or COBYLA could be used at both levels. The performance varies depending on the problem. For this specific problem, the timings look like this:

| Outer optimizer | Inner optimizer | Runtime |
| --------------- | --------------- | ------- |
| COBYLA          | SLSQP           | 2.2     |
| SLSQP           | COBYLA          | 0.6     |
| COBYLA          | COBYLA          | 1.1     |

Another option would be to use OpenMDAO’s pyOptSparseDriver which lets you use an optimizer such as IPOPT.