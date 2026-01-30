"""
.. _example_dpf_Boucher:

===================================
Example of depth potential function in slam
===================================
"""


# Authors: Julien Lefevre <julien.lefevre@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# Import of modules
import slam.curvature as sc
import slam.sulcal_depth as sdepth
import trimesh
import numpy as np
from scipy.spatial import Delaunay

#################################
# Define the example provided in Figure 5 of
# Depth potential function for folding pattern representation,
# registration and analysis
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
    faces_tri = Delaunay(np.vstack((X, Y)).T, qhull_options="QJ Qt Qbb")

    # Equation for Z
    M = params[0]
    sigma = params[1]  # called sigma_y in the paper
    Z = (
        M
        / sigma
        * np.exp(-(X**2) - Y**2 / (2 * sigma**2))
        * (Y**2 - sigma**2)
    )

    # Mesh
    coords = np.array([X, Y, Z]).transpose()
    mesh = trimesh.Trimesh(
        faces=faces_tri.simplices,
        vertices=coords,
        process=False)
    return mesh


params = [4, 0.25]
ax = 2
ay = 1
nstep = 50
mesh = boucher_surface(params, ax, ay, nstep)

##################################
# Compute dpf for various alpha

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
various_dpfs=sdepth.depth_potential_function(mesh, alphas=alphas)

amplitude_center = []
amplitude_peak = []
index_peak_pos = np.argmax(mesh.vertices[:, 2])
index_peak_neg = np.argmin(mesh.vertices[:, 2])
for i in range(len(various_dpfs)):
    amplitude_center.append(various_dpfs[i][index_peak_neg])
    amplitude_peak.append(various_dpfs[i][index_peak_pos])

####################################
# Fix alpha and vary M = params[0]
all_M = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
all_amplitudes = []

for M in all_M:
    mesh = boucher_surface([M, 0.25], ax, ay, nstep)
    dpfs = sdepth.depth_potential_function(
        mesh, alphas=[0.0015])
    all_amplitudes.append(dpfs[0][len(mesh.vertices) // 2])

#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################

import slam.plot as splt

vertices = mesh.vertices
# center the vertices
vertices = vertices - np.mean(vertices, axis=0)
vertices_translate = np.copy(vertices)
# rotate the vertices
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, vertices_translate.T).T
rot_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1],])
vertices_translate = np.dot(rot_z, vertices_translate.T).T

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'Boucher mesh alpha 0.001'
intensity_data = {}
intensity_data['values'] = various_dpfs[0]
intensity_data["mode"] = "vertex"
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'Boucher mesh alpha 100'
intensity_data = {}
intensity_data['values'] = various_dpfs[5]
intensity_data["mode"] = "vertex"
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

