"""
.. _example_distortion:

===================================
Example of morphological distortion in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# Importation of slam modules
import slam.distortion as sdst
import slam.differential_geometry as sdg
import slam.io as sio
import numpy as np

###############################################################################
# Loading an example mesh and a smoothed copy of it

mesh = sio.load_mesh("../examples/data/example_mesh.gii")
mesh_s = sdg.laplacian_mesh_smoothing(mesh, nb_iter=50, dt=0.1)

###############################################################################
# Computation of the angle difference between each faces of mesh and mesh_s
angle_diff = sdst.angle_difference(mesh, mesh_s)
angle_diff

###############################################################################
#
face_angle_dist = np.sum(np.abs(angle_diff), 1)
face_angle_dist

###############################################################################
# Computation of the area difference between each faces of mesh and mesh_s
area_diff = sdst.area_difference(mesh, mesh_s)
area_diff

###############################################################################
# Computation of the length difference between each edges of mesh and mesh_s
edge_diff = sdst.edge_length_difference(mesh, mesh_s)
edge_diff


#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################

import slam.plot as splt

# # Visualization of the original mesh
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
mesh_data['title'] = 'example_mesh.gii Original Mesh'
intensity_data = None
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

# ############################################################################
# # Visualization of the smoothed mesh

vertices = mesh_s.vertices
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
mesh_data['faces'] = mesh_s.faces
mesh_data['title'] = 'example_mesh.gii Smoothed Mesh'
intensity_data = None
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()
