# Installation tutorials

## Background
* In the OpenMDAO docs, the instructions are simply to use Anaconda and pip install.
* For people (undergraduate/graduate students, researchers, engineers in the industry, instructors, etc.) who are developing applications (i.e., beyond using example run scripts), this is not enough information.
* The pip-install approach may be somewhat sufficient for running self-contained example scripts, but once someone moves on to modifying the internals of OpenMDAO, OpenAeroStruct (OAS), Dymos, pyCycle, etc., it requires a complete reorientation, which there is no guidance for.
* I don't think I've ever used OpenMDAO or OAS in a project where at some point I didn't go in the source code and make modifications for debugging or development.
* I'm using OAS as an example here, but this could apply for other things (Dymos, pyCycle, etc.)
* For people who already have a lot on their plates (figuring out their physics/analysis, working on challenging design issues, juggling other project requirements, other courses, etc.) figuring out how to proceed on their own is quite prohibitive and frustrating (speaking as a grad student using openMDAO for 5 years and interacting with students as a teaching assistant and with external collaborators in industry and other universities).
* This is especially challenging for people who are new to the world of python/conda/git/pip/HPC/etc.
* Often I and my colleagues are not sure what the best practices are in the first place, and once we have figured out some workflow that works well, we have to repeatedly explain and help others debug or change their setups.

## Request: 
1) Can you create a more detailed set of instructions (or walkthrough videos) on how best to work with and set up OpenMDAO (and OpenAeroStruct etc.) on a windows machine, especially for people in the industry or undergraduates who typically have less OS freedom?
* For users who: maybe might develop vs. will never develop with these tools

(To clarify: what are recommended baseline setups? Below are a lot of options. My point is not that a solution should be available for each, but rather, because there are many degrees of freedom, it would help to have one recommended combination to go with to avoid issues, especially for new users. Of course, experienced users can go with whatever approach at their own risk and can get specific support from the stack overflow or by contacting you directly.)

* Anaconda prompt vs IDE (spyder, pycharm, etc) vs VScode vs etc...
* Links for more information on how to use: pip vs conda, a command prompt, etc. (depending on the recommendation)
* Links for learning more about python and especially OOP (recommend for people who will develop with OpenMDAO)
* Pip install vs git cloning?
* If pip install, where to edit OpenMDAO (and/or OpenAeroStruct/etc. code)
* If git cloning, cloning most recent vs checking out a particular version
* If git cloning, is version control an issue with firewall and other security infrastructure etc. (local vs remote version control)
* How to install/recompile after edits?
* What to do about frequent changes made to OpenMDAO? What is a good strategy to keep track? How to tell if updating is necessary? What to do if pip install changes versions of other codes? (in particular for new users)
* Seek feedback from the community during the hackathon.

2) Same for Macs

3) Same for Ubuntu

4) HPC
