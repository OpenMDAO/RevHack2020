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

There is one component that does the nested-optimization (i.e. no groups, connection, etc). Also the sub-optimization in this case uses finite differences for derivatives, and in the spirit of the problem given we assumed it would stay that way. So you wouldn't get the benefit of the analytic derivatives, and there is no other features of OpenMDAO being used in the sub-optimization... hence there isn't a lot of value in it. 
Also, there is just less code without the sub-problem. 

You can see how things line up against it in this case. If you wanted to switch to analytic derivatives, or if your sub-problem involved more analyses that were coupled together then you'd probably be better off with a sub-problem. 

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

## What about sequential optimization? 

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


# Our solutions: 
## Top Level Optimization Scripts 
* [Nested optimization with a sub-problem](./run_sequential_opt_using_subproblem.py)
* [Sequential Optimization with sub-problem](./run_sequential_opt_using_subproblem.py)

## Component implementations for the nested optimization
* [sub-problem](./components/compute_pitch_angles_using_subproblem.py)
* [Original solution given by John Jasa](./components/compute_pitch_angles.py)
* [Solver based approach](./components/compute_pitch_angles_solver.py)

## Optimizers Compatible with Nested Optimization

One consideration with nested optimizations is that some optimizers are not re-entrant. Re-entrant is a computer science term that refers to the ability to call a function while a previous call to that function has not been completed. This is a consideration when doing nested optimizations. If the optimizers were not written with this in mind, the results will be incorrect.

In particular, it is known that the current implementation of the SLSQP optimizer used in scipy’s minimize function is not re-entrant. Therefore, do not use SLSQP at multiple levels when using sub problems. ( Fortunately, there are implementations of SLSQP that are re-entrant, but unfortunately, they have not made their way into scipy. Here is an implementation from [Jacob Williams](https://github.com/jacobwilliams/slsqp)).

With OpenMDAO’s ScipyOptimizeDriver, one optimizer that is available and re-entrant is COBYLA. So the user could use SLSQP at one level and COBYLA at the other. Or COBYLA could be used at both levels. But using SLSQP at both levels will result in incorrect results.

For the code we provided in this solution, we used SLSQP at the top level and COBYLA in the sub problem.

There are no general guidelines on choosing which optimizer at each level. The performance varies depending on the problem. For this specific problem, the timings look like this, for example:

| Outer optimizer | Inner optimizer | Runtime |
| --------------- | --------------- | ------- |
| COBYLA          | SLSQP           | 2.2     |
| SLSQP           | COBYLA          | 0.6     |
| COBYLA          | COBYLA          | 1.1     |

Another option would be to use OpenMDAO’s pyOptSparseDriver which lets you use an optimizer such as IPOPT.


# A theoretical discussion on nested optimizations

## You can think of nested optimization as a form of Multi Disciplinary Feasible (MDF) architecture

The MDF architecture is typically thought of in terms of converging the governing equations for your analysis with a solver. 
That's an overly specific interpretation though. 
More generally, MDF removes degrees of freedom from the top level problem by handing them to a well behaved sub-solver. 

If that sub-solver happens to be a nonlinear solver and the degrees of freedom happen to be state variables for your analysis then you get the traditional view of MDF. 
But its equally valid to think of the sub-solver as an optimizer and the degrees of freedom as the operational variables for your problem. 

Generally speaking we know that the MDF solution approach tends to be slower but more reliably convergent, assuming you can reliably get a converged sub-problem for any design within the search space of the top level problem. 
It seems reasonable to trade some performance for improved stability to us! 
Just keep in mind that not every problem is well suited to an MDF style approach, and sometimes you can run into trouble if you can't get a reliably converged solution for your sub-problem. 

## Thinking of a sub-optimizer as a sub-solver

Regardless of what kind of optimizer you are using, one way to re-frame an optimization problem is to think of it as a nonlinear system. 
In early calculus lectures, we all learned that you can find the inflection point of a function by taking its derivatives and solving for when that goes to 0. 
Numerical optimizers are, broadly speaking, doing this exact same thing. 

Lets assume that you want to minimize a continuous function, `f(x)` without any constraints. 
From calculus we know that at that minimum point, `df_dx=0`. 
If you have some way to compute `df_fx` then you can 
a) use an optimizer 
b) treat `df_dx=0` as a residual and use a solver. 

Which is better? Well, using the optimizer might seem easier, especially because they almost always come with some way to internally approximate `df_dx` so you don't have to. 
However, for the case of nested optimization the solver approach is well worth considering. 
It offers greater numerical stability, and gives you the option of computing derivatives across the solved solution using OpenMDAO's analytic derivatives features. 

The first trick to making this work is that you need to somehow construct this residual equation from the derivative. 
The simplest way to do this is just to implement the finite-difference yourself: 

```python 

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
```

Then you can pass that as your residual function to a solver like this: 

```python 
root = brentq(fd_dpower__dpitch_angle, -15, 15, 
                          args=(wind_speed, drag_modifier))
```


### What about if you have constraints? then you need an optimizer, right? 
Nope! There are lots of approaches to handling constraints, but let me first say some scary words that you should look into if you are interested in this topic: Augmented Lagrangian, KKT conditions, Lagrange multipliers, slack variables, active set, barrier functions

Practically speaking, equality constraints are trivial. You just add them as additional residuals to be converged along with driving the derivative to 0. 
Inequality constraints are a bit trickier, but sometimes you can get away with a simple trick where you conditionally evaluate different equations for the same residual. 
There are some caveats, but this is a good tool to add to your toolbox. 

```python
def composite_residual(pitch_angle, wind_speed, drag_modifier, P_rated, return_power=False): 
    ''' a "trick" to apply a constraint is to conditionally evaluate different residuals''' 

    # NOTES: This trick works fine as long as one of two conditions is met: 
    # 1) Your residuals are c1 continuous across the conditional 
    # 2) Your residuals are c0 continuous and  you don't end up oscillating 
    #     back and forth across the breakpoint in the conditional

    power, d_power = fd_dpower__dpitch_angle(pitch_angle, wind_speed, drag_modifier)


    if power < -P_rated: 
        # NOTE: its usually beneficial to normalize your residuals by a reference quantity
        R = (power-P_rated)/P_rated
    else: 
        R = d_power

    if return_power: 
        return R, power
    else: 
        return R
```


