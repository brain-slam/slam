# slam
Surface anaLysis And Modeling

[![Build Status](https://travis-ci.org/gauzias/slam.svg?branch=master)](https://travis-ci.org/gauzias/slam) 
[![Coverage Status](https://coveralls.io/repos/github/gauzias/slam/badge.svg?branch=master)](https://coveralls.io/github/gauzias/slam?branch=master)

slam is a pure Python library for analysing and modeling surfaces represented as a triangular mesh.
It is an extension of Trimesh, which is an open source python module dedicated to general mesh processing:
https://github.com/mikedh/trimesh

Look at the doc!
https://gauzias.github.io/slam/

The present module will consist of extensions to adapt Trimesh for the purpose of surface analysis of brain MRI data.

## Visbrain is recommended for visualization

------------------
Installation:
------------------

1. install the lastest version of trimesh in easy (minimal dependency) mode:
    ```
    pip install trimesh[easy]
    ```
2. install visbrain (instead of pyglet):
    ```
    pip install visbrain
    ```
    for pyglet:
    ```
    conda install -c conda-forge pyglet
    ```
3. install nibabel
    ```
    pip install nibabel
    ```
4. install matplotlib
    ```
    conda install matplotlib
    ```
5. install gdist
    ```
    pip install cython
    pip install gdist
    ```
6. clone the current repo

7. try example scripts located in examples folder

------------------
Features (added value compared to Trimesh):
------------------

. io: read/write gifti (and nifti) file format 

. generate_parametric_surfaces: generation of parametric surfaces with random sampling

. geodesics: geodesic distance computation using gdist and networkx

. differential_geometry: several implementations of graph Laplacian (conformal, authalic, FEM...), texture Gradient

. mapping: several types of mapping between the mesh and a sphere, a disc...

. distortion: distortion measures between two meshes, for quantitative analysis of mapping properties

. remeshing: projection of vertex-level information between two meshes based on their spherical representation

. topology: mesh surgery (boundary indentification, hole closing will be added in the future)

. vertex_voronoi: compute the voronoi of each vertex of a mesh, usefull for numerous applications

. texture: a class to manage properly vertex-level information.

. plot: extension of pyglet and visbrain viewers to visualize slam objects


------------------
For contributors:
------------------

1. intall flake8 and autopep8
    ```
    pip install -U autopep8 flake8
   ```

2. install pytest
    ```bash
    pip install -U pytest 
    ```

