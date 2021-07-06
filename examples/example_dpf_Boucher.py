"""
.. _example_dpf_Boucher:

===================================
Example of depth potential function in slam
===================================
"""


# Authors: Julien Lefevre <julien.lefevre@univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Import of slam modules
import slam.distortion as sdst
import slam.differential_geometry as sdg
import slam.plot as splt
import trimesh
import numpy as np
from scipy.spatial import Delaunay


#################################
# Define the example provided in Figure 5 of
# Depth potential function for folding pattern representation, registration and analysis
# Maxime Boucher a,b, * , Sue Whitesides a , Alan Evans


def boucher_surface(params, ax, ay, nstep):
    # Parameters
    xmin, xmax = [-ax, ax]
    ymin, ymax = [-ay, ay]
    # Define the sampling
    stepx = (xmax - xmin) / nstep
    stepy = stepx * np.sqrt(3) / 2  # to ensure equilateral faces

    # Coordinates
    x = np.arange(xmin, xmax, stepx)
    y = np.arange(ymin, ymax, stepy)
    X, Y = np.meshgrid(x, y)

    X[::2] += stepx / 2
    X = X.flatten()
    Y = Y.flatten()

    # Delaunay
    faces_tri = Delaunay(np.vstack((X, Y)).T, qhull_options='QJ Qt Qbb')

    # Equation for Z
    M = params[0]
    sigma = params[1] # called sigma_y in the paper
    Z = M/sigma * np.exp(-X**2 - Y**2/(2*sigma**2)) * (Y**2 - sigma**2)

    # Mesh
    coords = np.array([X, Y, Z]).transpose()
    mesh = trimesh.Trimesh(faces=faces_tri.simplices, vertices=coords,
                                   process=False)
    return mesh


params = [4,0.25]
ax = 2
ay = 1
nstep = 50
mesh = boucher_surface(params, ax, ay, nstep)

##########################################################################
# Visualization of the mesh
visb_sc = splt.visbrain_plot(mesh=mesh, caption='Boucher mesh',bgcolor=[0.3,0.5,0.7])
visb_sc
visb_sc.preview()