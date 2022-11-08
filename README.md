# Surface anaLysis And Modeling (Slam)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![Build Status](https://travis-ci.org/gauzias/slam.svg?branch=master)](https://travis-ci.org/gauzias/slam) 
[![Coverage Status](https://coveralls.io/repos/github/gauzias/slam/badge.svg?branch=master)](https://coveralls.io/github/gauzias/slam?branch=master)

Slam is an open source python package dedicated to the representation of 
neuroanatomical surfaces stemming from MRI data in the form of triangular meshes and to their processing and analysis.
Slam is an extension of [Trimesh](https://github.com/mikedh/trimesh), an open source python package dedicated to triangular meshes processing.


## Main Features


   Look at the [doc](https://brain-slam.github.io/slam) for a complete overview of available features! 
   
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

## Prerequisites

 We highly recommend to rely on a (conda) virtual environment as provided by miniconda.
 See [miniconda installation instructions](https://docs.conda.io/en/latest/miniconda.html)  if you do not already have one. 
 Then create a virtual environment by typing the following lines in a terminal:
  ```
    conda create -q -n slam python=3.6
    conda activate slam
  ``` 
 This creates an empty conda virtual environment with Python 3.6 and basic packages
  (e.g. pip, setuptools) and make it the default python environment.


## User installation
``
pip install brain-slam
``

Then have a look at the [examples](https://brain-slam.github.
io/slam/auto_examples/index.html) from the doc website.
We will propose soon real tutorials dedicated to users.

## For developers / contributors

### Code of conduct
The very first thing to do before contributing is to read our 
[Code of conduct](CODE_OF_CONDUCT.md).

### Have a look at the github project!
We are using a github project to organize the code development and maintenance:
https://github.com/orgs/brain-slam/projects/1

If you are interested in contributing, please first have a look at it and contact us by creating a new issue.

### Developers installation
1. [Create an account](https://github.com/) on Github if you do not already have one
2. Sign in GitHub and fork  the [slam GitHub repository](https://github.com/brain-slam/slam)
3. Clone your personal slam fork in your current local directory
    ```# replace <username> by your Github login 
    git clone https://github.com/<username>/slam
    ```
4. Perform a full slam installation in editable mode
   ```
    pip install -e .['dev']
   ```
5. Set upstream repository to keep your clone up-to-date
   ```
    git remote add upstream https://github.com/brain-slam/slam.git
   ```
You are now ready to modify slam code and submit a pull request

## Dependencies 
These dependencies, whether mandatory or optional, are managed automatically and transparently for the user during the installation phase and are listed here for the sake of completeness.

### Mandatory
In order to work fine, slam requires:

+ a Python 3.6 installation 

+ setuptools

+ pip
 
+  numpy

+  scipy

+  cython

+  trimesh

+  nibabel


    
### Distance computation (Optional)


tvb-gdist is recommended for geodesic distance/shortest paths computations


   

## Hall of fame

All contributions are of course much welcome!
In addition to the global thank you to all contributors to this project, a special big thanks to:

. https://github.com/alexpron and https://github.com/davidmeunier79 for their precious help for setting up continuous integration tools.

. https://github.com/EtienneCmb for his help regarding visualization and Visbrain (https://github.com/EtienneCmb/visbrain).

. https://github.com/aymanesouani for his implementation of a very nice curvature estimation technique.

. https://github.com/Anthys for implementing the curvature decomposition and many unitests

.  to be continued...




## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://alexpron.github.io/"><img src="https://avatars0.githubusercontent.com/u/45215023?v=4" width="100px;" alt=""/><br /><sub><b>alexpron</b></sub></a><br /><a href="#maintenance-alexpron" title="Maintenance">🚧</a> <a href="#projectManagement-alexpron" title="Project Management">📆</a> <a href="#ideas-alexpron" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/gauzias/slam/commits?author=alexpron" title="Code">💻</a></td>
    <td align="center"><a href="https://sites.google.com/site/julienlefevreperso/"><img src="https://avatars2.githubusercontent.com/u/19426328?v=4" width="100px;" alt=""/><br /><sub><b>JulienLefevreMars</b></sub></a><br /><a href="https://github.com/gauzias/slam/commits?author=JulienLefevreMars" title="Code">💻</a> <a href="https://github.com/gauzias/slam/commits?author=JulienLefevreMars" title="Documentation">📖</a> <a href="#example-JulienLefevreMars" title="Examples">💡</a> <a href="#ideas-JulienLefevreMars" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/gauzias/slam/commits?author=JulienLefevreMars" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/tianqisong0117"><img src="https://avatars2.githubusercontent.com/u/47243851?v=4" width="100px;" alt=""/><br /><sub><b>Tianqi SONG</b></sub></a><br /><a href="https://github.com/gauzias/slam/commits?author=tianqisong0117" title="Code">💻</a> <a href="#example-tianqisong0117" title="Examples">💡</a> <a href="#ideas-tianqisong0117" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/EtienneCmb"><img src="https://avatars3.githubusercontent.com/u/15892073?v=4" width="100px;" alt=""/><br /><sub><b>Etienne Combrisson</b></sub></a><br /><a href="https://github.com/gauzias/slam/commits?author=EtienneCmb" title="Code">💻</a> <a href="#tool-EtienneCmb" title="Tools">🔧</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
