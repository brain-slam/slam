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

1. install the lastest version of trimesh in easy (minimal dependency) mode:
    ```
    pip install trimesh[easy]
    ```
2. install nibabel
    ```
    conda install nibabel
    ```
3. install matplotlib
    ```
    conda install matplotlib
    ```
4. install gdist
    ```
    pip install cython
    pip install gdist
    ```
5. clone the current repo

6. try example scripts located in examples folder


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

