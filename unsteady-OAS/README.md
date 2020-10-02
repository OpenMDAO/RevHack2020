# How to carry out an unsteady VLM analysis with OpenAeroStruct?

## Background:
* OpenAeroStruct (OAS) is gaining popularity and is often advertized as a good example of a practical application that takes advantage of the strengths of OpenMDAO.
* github.com/mdolab/openaerostruct
* It is difficult to figure out how to carry out an unsteady VLM analysis with OAS, given that something like OAS is already so complex.
* However, for many kinds of analyses, unsteady simulations are necessary.
* A few years ago, Giovanni Pesare implemented an unsteady VLM solver with OAS with a little direction from me (see attached thesis).
* The only approach I could think of at the time was to instantiate an OAS group for each time step.
* I still don't know what the best/correct way to implement this is.

## Request:
* Implement an unsteady VLM analysis with OAS (e.g., see the attached thesis)
