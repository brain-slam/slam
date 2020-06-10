# Surface anaLysis And Modeling (Slam)

[![Build Status](https://travis-ci.org/gauzias/slam.svg?branch=master)](https://travis-ci.org/gauzias/slam) 
[![Coverage Status](https://coveralls.io/repos/github/gauzias/slam/badge.svg?branch=master)](https://coveralls.io/github/gauzias/slam?branch=master)

Slam is an open source python package dedicated to the representation of neuroanatomical surfaces stemming from MRI data in the form of triangular meshes and to their processing and analysis.
Slam is an extension of [Trimesh](https://github.com/mikedh/trimesh), an open source python package dedicated to triangular meshes processing.


## Main Features


   Look at the [doc](https://gauzias.github.io/slam) for a complete overview of available features! 
   
+ ``io``: read/write gifti (and nifti) file format 

+ ``generate_parametric_surfaces``: generation of parametric surfaces with random sampling

+ ``geodesics``: geodesic distance computation using tvb-gdist and networkx

+ ``differential_geometry``: several implementations of graph Laplacian (conformal, authalic, FEM...), texture Gradient

+ ``mapping``: several types of mapping between the mesh and a sphere, a disc...

+ ``distortion``: distortion measures between two meshes, for quantitative analysis of mapping properties

+ ``remeshing``: projection of vertex-level information between two meshes based on their spherical representation

+ ``topology``: mesh surgery (boundary indentification, large hole closing)

+ ``vertex_voronoi``: compute the voronoi of each vertex of a mesh, usefull for numerous applications

+ ``texture``: a class to manage properly vertex-level information.

+ ``plot``: extension of pyglet and visbrain viewers to visualize slam objects

## Dependencies 
These dependencies, whether mandatory or optional, are managed automatically and transparently for the user during the installation phase and are listed here for the sake of completeness. However, the user must have:

+ a Python3.6 installation 

+ setuptools

+ pip
 
### Mandatory
In order to work fine, slam requires:

+  numpy

+  scipy

+  cython

+  trimesh

+  nibabel

+  matplotlib
    
### Optional

#### Distance computation

tvb-gdist is recommended for geodesic distance/shortest paths computations

#### Visualisation 

visbrain is highly recommended for visualisation see (https://github.com/EtienneCmb/visbrain)

#### Developers

+  flake8, autopep8

+ pytest, pytest-cov

+ codecov
   
## Installation:

### Users

1. Clone the current repository

2. Move to slam folder and type one of the two following commands in terminal
    ```
    python setup.py install
    pip install . 
    ```
3. Try example scripts located in ``examples`` folder


### Contributors

 * install default packages as well as syntax validation ( ``flake8``, ``autopep8``) and test (``pytest``, ``pytest-cov``) packages
    ```
    pip install -e .['default-dev]
    ```

## Hall of fame

All contributions are of course much welcome!
In addition to the global thank you to all contributors to this project, a special big thanks to:

. https://github.com/alexpron and https://github.com/davidmeunier79 for their precious help for setting up continuous integration tools.

. https://github.com/EtienneCmb for his help regarding visualization and Visbrain (https://github.com/EtienneCmb/visbrain).

. https://github.com/aymanesouani for his implementation of a very nice curvature estimation technique.

.  to be continued...



