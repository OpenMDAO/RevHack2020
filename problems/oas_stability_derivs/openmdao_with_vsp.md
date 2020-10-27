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

### Install OpenMDAO
 In your Python environment, run `pip install 'openmdao[all]'`

### Sample Code
This is an OpenMDAO component wrapper for a VSP model of a modified CRM (eCRM-001) that is used to demonstrate the computation and use of stability derivatives in a design problem. This code is based on a sample provided used during [OpenMDAO RevHack 2020](https://github.com/OpenMDAO/RevHack2020).

Required modules:
```
import pickle
import itertools

import numpy as np

import openmdao.api as om

import openvsp as vsp
import degen_geom
```

Create a new subclass based on OpenMDAO [ExplicitComponent](http://openmdao.org/twodocs/versions/latest/features/core_features/defining_components/explicitcomp.html). Set some [options](http://openmdao.org/twodocs/versions/latest/features/core_features/defining_components/options.html) with the names of the geometries in the VSP project that we want to work with. The `reduced` option reduces processing time for testing purposes.
```
class VSPeCRM(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('horiz_tail_name', default='Tail',
                             desc="Name of the horizontal tail in the vsp model.")
        self.options.declare('vert_tail_name', default='VerticalTail',
                             desc="Name of the vertical tail in the vsp model.")
        self.options.declare('wing_name', default='Wing',
                             desc="Name of the wing in the vsp model.")
        self.options.declare('reduced', default=False,
                             desc="When True, output reduced meshes instead of full-size ones. "
                             "Running with a smaller mesh is of value when debugging.")
```

Load the OpenVSP project from a VSP3 file and find the IDs of the relevant geometries.
```
    def setup(self):
        options = self.options
        horiz_tail_name = options['horiz_tail_name']
        vert_tail_name = options['vert_tail_name']
        wing_name = options['wing_name']
        reduced = options['reduced']

        # Read the geometry.
        vsp_file = 'eCRM-001.1_wing_tail.vsp3'
        vsp.ReadVSPFile(vsp_file)

        self.wing_id = vsp.FindGeomsWithName(wing_name)[0]
        self.horiz_tail_id = vsp.FindGeomsWithName(horiz_tail_name)[0]
        self.vert_tail_id = vsp.FindGeomsWithName(vert_tail_name)[0]
```

Set up inputs with initial values, and outputs with units and 3-dimensional shapes. The finite difference approximation method for partial derivatives is used.
```
        self.add_input('wing_cord', val=59.05128,)
        self.add_input('vert_tail_area', val=2295.)
        self.add_input('horiz_tail_area', val=6336.)

        # Shapes are pre-determined.
        if reduced:
            self.add_output('wing_mesh', shape=(12, 9, 3), units='inch')
            self.add_output('vert_tail_mesh', shape=(9, 9, 3), units='inch')
            self.add_output('horiz_tail_mesh', shape=(9, 9, 3), units='inch')
        else:
            self.add_output('wing_mesh', shape=(23, 33, 3), units='inch')
            self.add_output('vert_tail_mesh', shape=(33, 9, 3), units='inch')
            self.add_output('horiz_tail_mesh', shape=(33, 9, 3), units='inch')

        self.declare_partials(of='*', wrt='*', method='fd')
```

Compute the OpenVSP meshes. Using the previously located geometries, set the values of the VSP parameters to the initial input values from `setup()`. Then, reset the OpenVSP model to a blank slate with `Update()`.
```
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # set values
        vsp.SetParmVal(self.vert_tail_id, "TotalArea", "WingGeom", inputs['vert_tail_area'][0])
        vsp.SetParmVal(self.horiz_tail_id, "TotalArea", "WingGeom", inputs['horiz_tail_area'][0])
        vsp.SetParmVal(self.wing_id, "TotalChord", "WingGeom", inputs['wing_cord'][0])

        vsp.Update()
        vsp.Update()  # just in case..
```

Compute the degenerate geometry representation for the OpenVSP components, and obtain the measurements.
```
        # run degen geom to get measurements
        dg:degen_geom.DegenGeomMgr = vsp.run_degen_geom(set_index=vsp.SET_ALL)
        obj_dict = {p.name:p for p in dg.get_all_objs()}

        # pull measurements out of degen_geom api
        degen_obj: degen_geom.DegenGeom = obj_dict[self.options['wing_name']]
        wing_pts = self.vsp_to_point_cloud(degen_obj)

        degen_obj: degen_geom.DegenGeom = obj_dict[self.options['horiz_tail_name']]
        horiz_tail_pts = self.vsp_to_point_cloud(degen_obj)

        degen_obj: degen_geom.DegenGeom = obj_dict[self.options['vert_tail_name']]
        vert_tail_pts = self.vsp_to_point_cloud(degen_obj)
```

Change the shape of the point cloud for use with OpenAeroStuct, using Fortran-like index order. Since the symmetry points are duplicated, slice the array in half.
```
        # OAS expects x stripes.
        wing_pts = wing_pts.reshape((45, 33, 3), order='F')
        horiz_tail_pts = horiz_tail_pts.reshape((33, 9, 3), order='F')
        vert_tail_pts = vert_tail_pts.reshape((33, 9, 3), order='F')

        # Meshes have symmetry pts duplicated (not mirrored.) Use half.
        wing_pts = wing_pts[:23, :, :]
        horiz_tail_pts = horiz_tail_pts[:17, :, :]
        vert_tail_pts = vert_tail_pts[:17, :, :]

        # Reduce for testing. (See John Jasa's recommendations in the docs.)
        if self.options['reduced']:
            wing_pts = wing_pts[::2, ::4, :]
            horiz_tail_pts = horiz_tail_pts[::2, :, :]
            vert_tail_pts = vert_tail_pts[::2, :, :]

        # Flip around so that FEM normals yield positive areas.
        wing_pts = wing_pts[::-1, ::-1, :]
        horiz_tail_pts = horiz_tail_pts[::-1, ::-1, :]
        vert_tail_pts = vert_tail_pts[::-1, ::-1, :]

        # outputs go here
        outputs['wing_mesh'] = wing_pts
        outputs['vert_tail_mesh'] = horiz_tail_pts
        outputs['horiz_tail_mesh'] = vert_tail_pts
```

Convert an OpenVSP degenerate geometry to a NumPy N-dimensional array of points using [itertools](https://docs.python.org/3/library/itertools.html).
```
    def vsp_to_point_cloud(self, degen_obj: degen_geom.DegenGeom)->np.ndarray:
        npts = degen_obj.surf.num_pnts
        n_xsecs = degen_obj.surf.num_secs

        points = np.empty((npts * n_xsecs, 3))
        points[:, 0] = list(itertools.chain.from_iterable(degen_obj.surf.x))
        points[:, 1] = list(itertools.chain.from_iterable(degen_obj.surf.y))
        points[:, 2] = list(itertools.chain.from_iterable(degen_obj.surf.z))

        return points
```

Set up and run the model.
```
if __name__ == "__main__":

    vsp_comp = VSPeCRM(horiz_tail_name="Tail",
                       vert_tail_name="VerticalTail",
                       wing_name="Wing",
                       reduced=True)

    p = om.Problem()

    model = p.model

    p.model.add_subsystem("vsp_comp", vsp_comp)

    p.setup()

    p.run_model()
```

Serialize and save the data for later use.
```
    data = {}
    for item in ['wing_mesh', 'vert_tail_mesh', 'horiz_tail_mesh']:
        data[item] = p.get_val(f"vsp_comp.{item}", units='m')

    with open('baseline_meshes_reduced.pkl', 'wb') as f:
        pickle.dump(data, f)

    print('done')
```
