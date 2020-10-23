# Creating an OpenMDAO Component Wrapper for a VSP Model
Here are some lessons we learned from installing OpenVSP on various platforms and using an OpenMDAO component to interface with a VSP model.

## Installing OpenVSP and its Python Packages
The OpenVSP installation process is fairly well documented, but we found some steps that weren't completely described and ran into different issues on different platforms such as MacOS and CentOS.

If you're running Debian or Ubuntu, the [provided installation instructions](http://openvsp.org/wiki/doku.php?id=ubuntu_instructions) work well for the application portion; follow those and skip to step 7 below. Steps 1-6 are adapted here for MacOS and CentOS 7.

### 1. Install dependencies.
 - On MacOS, an easy way to install these is with a package manager such [Homebrew](https://brew.sh/).
 - Chief among these is CMake. Make sure it's at least version 3, or some functions will fail.
 - MacOS with Homebrew: `brew install graphviz doxygen libjpeg cmake gfortran`
 - CentOS 7: `sudo yum install -y graphviz doxygen libjpeg cmake3 gfortran glew glew-devel fltk fltk-devel fltk-fluid libxml2-static libxml2-devel`

### 2. Set up and activate a Python virtual environment.
 - [Anaconda](https://www.anaconda.com) or Python's [venv](https://docs.python.org/3/tutorial/venv.html) are options.
 - If swig isn't installed, run `pip install swig`.
 - Swig may automatically select the system Python version rather than the one in your environment. The following with prevent this:
 ```
 export PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"`
 export PYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")`
 ```

### 3. Obtain OpenVSP source code.
Clone the [OpenVSP GitHub repository](https://github.com/OpenVSP/OpenVSP.git):
```
mkdir OpenVSP; cd OpenVSP
mkdir repo build buildlibs
git clone --depth=1 https://github.com/OpenVSP/OpenVSP.git repo
```

### 4. Build the libraries.
 - MacOS:
 ```
 cd buildlibs
 cmake -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} ../repo/Libraries -DCMAKE_BUILD_TYPE=Release
 make -j8
 ```

 - CentOS 7 (we ran into some version conflicts building the GUI on CentOS):
 ```
 cd buildlibs
 cmake3 -DVSP_NO_GRAPHICS=true -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DVSP_USE_SYSTEM_LIBXML2=true      -DVSP_USE_SYSTEM_FLTK=true ../repo/Libraries -DCMAKE_BUILD_TYPE=Release
 make -j8
 ```

### 5. Build the OpenVSP application.
 - MacOS:
 ```
 cd ..
 export BUILD_LIBS_PATH=`pwd`
 cd build
 cmake ../repo/src/ -DVSP_LIBRARY_PATH=${BUILD_LIBS_PATH}/buildlibs \
	   -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} \
       -DCMAKE_BUILD_TYPE=Release
 make -j8
 ```
 - CentOS 7:
 ```
 cd ..
 export BUILD_LIBS_PATH=`pwd`
 cd build
 cmake3 ../repo/src/ \
        -DVSP_NO_GRAPHICS=true -DVSP_LIBRARY_PATH=${BUILD_LIBS_PATH}/buildlibs \
        -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} \
        -DCMAKE_BUILD_TYPE=Release
 make -j8
 ```

### 6. Make a folder and zip file with binaries.
 This step is important because you'll need this directory structure to install the Python packages:
 - `make package`

### 7. Set the installation prefix and OpenVSP version.
 You'll need to determine from the repository which version of OpenVSP you're working with.
 - With Anaconda: `export INST_PREFIX=$CONDA_PREFIX OPENVSP_VERSION=X.Y.Z`
 - With venv: `export INST_PREFIX=$VIRTUAL_ENV OPENVSP_VERSION=X.Y.Z`
 - Otherwise, set to a location that works for you: `export INST_PREFIX=$HOME/opt OPENVSP_VERSION=X.Y.Z`

### 8. Install the OpenVSP Python packages.
 - MacOS:
 ```
 # Set to the OpenVSP version you're installing:
 pushd _CPack_Packages/MacOS/ZIP/OpenVSP-${VERSION}-MacOS/python
 pip install -r requirements.txt
 pushd ..
 cp vspaero vspscript vspslicer vspviewer $INST_PREFIX/bin
 popd 
 popd
 ```
 - CentOS 7:
 ```
 # Set to the OpenVSP version you're installing:
 pushd _CPack_Packages/Linux/ZIP/OpenVSP-${VERSION}-Linux/python
 pip install -r requirements.txt
 pushd ..
 cp vspaero vspscript vspslicer $INST_PREFIX/bin
 popd 
 popd
 ```

### 9. Test
 A quick test to make sure the installation is in place: `python -c "import openvsp"`

## Incorporating OpenVSP into OpenMDAO
