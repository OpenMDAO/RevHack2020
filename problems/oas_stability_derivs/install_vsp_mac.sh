#!/bin/bash
VERSION=3.21.2
# MANUAL PREREQUISITES:
# Homebrew
which brew > /dev/null 2>&1 || {
	Homebrew not found, obtain from https://brew.sh/
	exit 1
}

# Activate the Python environment you want to work from; pip should be available
which pip > /dev/null 2>&1 || {
	Python pip not found, please install/activate an environment first.
	exit 1
}

if [ -n "$CONDA_PREFIX" ]; then
	INST_PREFIX=$CONDA_PREFIX
elif
   [ -n "$VIRTUAL_ENV" ]; then
	INST_PREFIX=$VIRTUAL_ENV
else
	INST_PREFIX=$HOME/opt
fi

set -e
trap 'cmd_failed $? $LINENO' EXIT

cmd_failed() {
        if [ "$1" != "0" ]; then
                echo "FATAL ERROR: The command failed with error $1 at line $2."
                exit 1
        fi
}

# Install dependencies:
brew install graphviz doxygen libjpeg git-gui cmake gfortran
brew cask install basictex
which swig > /dev/null 2>&1 || pip install swig

# Make main directory and clone OpenVSP source
mkdir -p OpenVSP; cd OpenVSP
mkdir -p repo build buildlibs
[ ! -d repo/.git ] && git clone -b OpenVSP_${VERSION} --depth=1 https://github.com/OpenVSP/OpenVSP.git repo

# Prepare build files for the libraries and build them:
cd buildlibs
cmake	-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
	../repo/Libraries -DCMAKE_BUILD_TYPE=Release
make -j8

# Set up and build OpenVSP:
cd ..
BUILD_LIBS_PATH=`pwd`
cd build
cmake ../repo/src/ \
	-DVSP_LIBRARY_PATH=${BUILD_LIBS_PATH}/buildlibs \
	-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
	-DCMAKE_BUILD_TYPE=Release
make -j8

# Make a zip file:
make package

# Install OpenVSP Python Packages
CHARM_BUILD=CHARM/charm/charm_fortran_utilities/build
pushd _CPack_Packages/MacOS/ZIP/OpenVSP-${VERSION}-MacOS/python
mkdir -p $CHARM_BUILD
pushd $CHARM_BUILD
gfortran -o bg2charm_thick ../src/bg2charm_thick.f
popd

# If you are not going to modify the packages:
pip install -r requirements.txt
# If you want to modify the python packages
# pip install -r requirements-dev.txt
pushd ..
cp vspaero vspscript vspslicer vspviewer $INST_PREFIX/bin
popd 
popd
python -c 'import openvsp' || {
	echo "OpenVSP Python Packages did not install correctly, cannot import openvsp"
	exit 1
}
