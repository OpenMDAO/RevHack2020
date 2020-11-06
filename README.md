
# OpenMDAO RevHack 2020

RevHack is a reverse hackathon where the OM users propose problems to be solved and then the dev team tries to solve them! 
The primary dev team goal is to gain a better understanding of the use cases that users are struggling with, and to learn to see our framework through your eyes. 

What you get out of it is two-fold: 
1) We solves some problems for you 
2) We go into detail about our thought process and justifications for how we solved them, so you can learn something about how we see things. 
 
## Problem Solutions

### The ones we finished 
* OpenAeroStruct+VSP to optimize an aircraft subject stability constraints
* [eVTOL takeoff optimization with Dymos][prob-evtol]
* [Nested optimizations][prob-nested]
* [Optimize an OpenMDAO model with CMA-ES optimizer][prob-cma_es]
* [Unsteady VLM simulation][prob-unsteady-vlm]

### The ones we didn't get to
* OpenAeroStruct+Dymos for aircraft trajectory optimization
* Integrate an OpenAeroStruct analysis into a Dymos trajectory analysis model (@shamsheersc19)

## General solution approaches: 

Over the course of solving the problems we noticed a few general themes that were worth discussing in more detail.
These topics involve general model building practices and concepts, and don't deal with specific solutions. 

* [Building an OpenVSP Component][openvsp] -- Google Collab workbook that builds an OpenVSP component
* [You don't have to use a driver!][no-driver] -- Exploit the parts you like, leave the parts you don't! 
* [Sub-problems][subproblem] -- they are pretty handy in some situations! 
* [How big should I make my components?][how-big] --- How many lines of code should I aim for? 
* [Unsteady/transient analysis in OpenMDAO][unsteady] --- The is a right way, and some other right ways!


[no-driver]: ./solution_approaches/no_driver.md
[subproblem]: ./solution_approaches/sub_problems.md
[unsteady]: ./solution_approaches/unsteady_analysis.md
[how-big]: ./solution_approaches/how_big.md
[openvsp]: ./solution_approaches/OpenVSP_OpenMDAO/OpenVSPwithOpenMDAO.ipynb

[prob-evtol]: ./problems/evtol_trajectory
[prob-nested]: ./problems/nested_optimization
[prob-unsteady-vlm]: ./problems/unsteady_vlm
[prob-cma_es]: ./problems/cma_es