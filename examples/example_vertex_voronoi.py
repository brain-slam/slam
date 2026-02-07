"""
.. _example_vertex_voronoi:

===================================
Vertex voronoi example in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Importation of slam modules
import slam.io as sio
import slam.vertex_voronoi as svv
import numpy as np


###############################################################################
# Load a mesh
mesh = sio.load_mesh("../examples/data/example_mesh.gii")

###############################################################################
# Compute the vertex voronoi
vert_vor = svv.vertex_voronoi(mesh)

###############################################################################
# vertex voronoi is a numpy array of the same size as mesh.vertices
print(mesh.vertices.shape)
print(vert_vor.shape)

###############################################################################
# vertex voronoi corresponds to the "area of vertices".
# It is the sum of 1/3 of the triangles to which the vertex belongs.
# So the sum of vertex voronoi is equal to the area of the entire mesh
print(np.sum(vert_vor) - mesh.area)


#############################################################################
# VISUALIZATION USING plotly
#############################################################################

import slam.plot as splt
###############################################################################
# Visualization

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'example_mesh.gii'

intensity_data = {}
intensity_data['values'] = vert_vor
intensity_data["mode"] = "vertex"
Fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()
