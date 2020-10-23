# Creating an OpenMDAO Component Wrapper for a VSP Model
Lessons learned from installing OpenVSP on various platforms and using an OpenMDAO component to interface with a VSP model.

## Installing OpenVSP and its Python Packages
The OpenVSP installation process is fairly well documented, but we found some steps that weren't completely described and ran into different issues on different platforms such as MacOS and CentOS.

If you're running Debian or Ubuntu, the [provided installation instructions](http://openvsp.org/wiki/doku.php?id=ubuntu_instructions) work well for the application portion.

1. Install dependencies. On MacOS, an easy way to install these is with a package manager such [Homebrew](https://brew.sh/).
 - Chief among these is CMake. Make sure it's at least version 3, or some functions will fail.
 - MacOS with Homebrew: `brew install graphviz doxygen libjpeg cmake gfortran`
 - CentOS 7: `sudo yum install -y graphviz doxygen libjpeg cmake3 gfortran glew glew-devel fltk fltk-devel fltk-fluid libxml2-static libxml2-devel`

2. Set up a Python virtual environment using Anaconda or Python's venv.
 - If swig isn't installed, run `pip install swig`.
 - swig may automatically select the system Python version rather than the one in your environment. To prevent this:
   `export PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"`
   `export PYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")`

Obtain OpenVSP source code by cloning the [OpenVSP GitHub repository](https://github.com/OpenVSP/OpenVSP.git).



## Incorporating OpenVSP into OpenMDAO