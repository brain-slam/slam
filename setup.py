#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

BASE_REQUIREMENTS = ["numpy", "trimesh", "nibabel"]
TEST_REQUIREMENTS = ['flake8', 'autopep8', 'pytest','pytest-cov', 'codecov']
GDIST = ['tvb-gdist']
VISU = ["matplotlib", "visbrain"]

setup(
    name="slam",
    version="0.0.1",
    author="Guillaume Auzias",
    description="Surface anaLysis And Modeling",
    url="https://github.com/gauzias/slam",
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',                     # enforce Python 3.6 as minimum
    install_requires=BASE_REQUIREMENTS,
    extras_require={
        'default-dev': TEST_REQUIREMENTS,
        'advanced-user': GDIST,
        'advanced-dev':  GDIST + TEST_REQUIREMENTS,
        'full': TEST_REQUIREMENTS + VISU
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix", "Operating System :: MacOS :: MacOS X"
    ],


)

