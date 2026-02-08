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

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'Original Mesh'
intensity_data = None
fig1 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
fig1.show()
fig1

# ############################################################################
# # Visualization of the smoothed mesh

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = mesh_s.vertices
mesh_data['faces'] = mesh_s.faces
mesh_data['title'] = 'Smoothed Mesh'
intensity_data = None
fig2 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
fig2.show()
