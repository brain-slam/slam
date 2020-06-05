#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

setup(
    name="brain-slam",
    version='0.0.1',
    packages=find_packages(),
    author="Guillaume Auzias",
    description="Surface anaLysis And Modeling",
    license='MIT',
    install_requires=["numpy", "trimesh", "matplotlib", "nibabel", "cython", "gdist"]
)

