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
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# Importation of slam modules
import slam.io as sio
import slam.vertex_voronoi as svv
import numpy as np


###############################################################################
mesh = sio.load_mesh("../examples/data/example_mesh.gii")
mesh.apply_transform(mesh.principal_inertia_transform)

###############################################################################
vert_vor = svv.vertex_voronoi(mesh)
print(mesh.vertices.shape)
print(vert_vor.shape)
print(np.sum(vert_vor) - mesh.area)


#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################

import slam.plot as splt
###############################################################################
# Visualization

vertices = mesh.vertices
vertices = vertices - np.mean(vertices, axis=0)

vertices_translate = np.copy(vertices)
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, vertices_translate.T).T

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'example_mesh.gii'

intensity_data = {}
intensity_data['values'] = vert_vor
intensity_data["mode"] = "vertex"
visb_sc = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
visb_sc.show()
