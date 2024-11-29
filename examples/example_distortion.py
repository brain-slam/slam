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
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# # Visualization of the original mesh
# visb_sc = splt.visbrain_plot(mesh=mesh, caption="original mesh")
# visb_sc.preview()
# ############################################################################
# # Visualization of the smoothed mesh
# visb_sc = splt.visbrain_plot(mesh=mesh_s, caption="smoothed mesh")
# visb_sc.preview()
