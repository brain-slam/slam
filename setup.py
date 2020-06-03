#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

GDIST = ['tvb-gdist']
TEST_PKGS = ['flake8', 'autopep8', 'pytest', 'codecov']

setup(
    name="slam",
    version_config={
        "version_format": "{tag}.dev{sha}",    # automatically retrieve version from git tag
        "starting_version": "0.0.1"            # defaut version if no tag provided
    },
    setup_requires=['better-setuptools-git-version'],
    author="Guillaume Auzias",
    description="Surface anaLysis And Modeling",
    url="https://github.com/gauzias/slam",
    license='MIT',
    packages=find_packages(),
    python_requires='==3.6',                     # enforce Python 3.6 as default version
    install_requires=["numpy", "trimesh", "matplotlib", "nibabel", "cython"],
    extras_require={
        'gdist': GDIST,
        'test': TEST_PKGS,
        'full': GDIST + TEST_PKGS
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix", "Operating System :: MacOS :: MacOS X"
    ],


)

