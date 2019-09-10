# slam
Surface anaLysis And Modeling

[![Build Status](https://travis-ci.org/gauzias/slam.svg?branch=master)](https://travis-ci.org/gauzias/slam) 
[![Coverage Status](https://coveralls.io/repos/github/gauzias/slam/badge.svg?branch=master)](https://coveralls.io/github/gauzias/slam?branch=master)

slam is a pure Python library for analysing and modeling surfaces represented as a triangular mesh.
It is an extension of Trimesh, which is an open source python module dedicated to general mesh processing:
https://github.com/mikedh/trimesh

The present module will consist of extensions to adapt Trimesh for the purpose of surface analysis of brain MRI data.

------------------
Installation:
------------------

-install the lastest version of trimesh in easy (minimal dependency) mode:

pip install trimesh[easy]

-install nibabel

conda install nibabel

-install pyglet

conda install -c conda-forge pyglet

-install matplotlib

conda install matplotlib

-install gdist

pip install cython

pip install gdist

-clone the current repo

-try example scripts located in examples folder


------------------
For contributors:
------------------

-intall flake8 and autopep8

pip install -U autopep8 flake8

-install pytest

pip install -U pytest 
