"""
.. _example_curvature:

===================================
example of curvature estimation in slam
===================================
"""

# Authors: Julien Lefevre <julien.lefevre@univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# importation of slam modules
import slam.utils as ut
import numpy as np
from scipy.spatial import Delaunay
import trimesh
import slam.plot as splt
import slam.curvature as scurv

###############################################################################
# Function to generate 3-4-...n hinge
# Parameters


def generate_hinge(n_hinge=3, n_step=50, m=-1/5, M=1/5, regularity = 'regular'):
    xmin, xmax = [m, M]
    ymin, ymax = [m, M]

    # Coordinates
    x = np.linspace(xmin, xmax, n_step)
    y = np.linspace(ymin, ymax, n_step)
    X, Y = np.meshgrid(x, y)

    # Delaunay triangulation
    X = X.flatten()
    Y = Y.flatten()
    faces_tri = Delaunay(np.vstack((X, Y)).T, qhull_options='QJ Qt Qbb')

    # Equation

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, phi

    R, Phi = cart2pol(X, Y)

    if regularity == 'regular':
        alpha_min = 2
        alpha_max = 4
        amplitude = 3
    else:
        alpha_min = 1/2
        alpha_max = 2
        amplitude = 0.5

    exponent = (alpha_max - alpha_min)/2 * np.cos(n_hinge * Phi) + \
               (alpha_min + alpha_max)/2
    Z = - amplitude * R ** exponent
    coords = np.stack([X, Y, Z])
    mesh = trimesh.Trimesh(faces=faces_tri.simplices, vertices=coords.T, process=False)
    return mesh


###############################################################################
# Creating an examplar mesh

mesh = generate_hinge(n_hinge = 4)
mesh_curvatures = scurv.curvatures_and_derivatives(mesh)
mean_curvature = 1/2* mesh_curvatures[0].sum(axis=0)
scene = splt.pyglet_plot(mesh, mean_curvature, plot_colormap=False, background_color=[255] * 4)
