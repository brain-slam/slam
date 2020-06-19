"""
.. _example_vertex_voronoi:

===================================
Vertex voronoi example in slam
===================================
"""

# Authors: Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Importation of slam modules
import slam.io as sio
import slam.plot as splt
import slam.vertex_voronoi as svv
import numpy as np


###############################################################################
#
mesh = sio.load_mesh('data/example_mesh.gii')
mesh.apply_transform(mesh.principal_inertia_transform)

###############################################################################
#
vert_vor = svv.vertex_voronoi(mesh)
print(mesh.vertices.shape)
print(vert_vor.shape)
print(np.sum(vert_vor) - mesh.area)

###############################################################################
# Visualization
visb_sc = splt.visbrain_plot(mesh=mesh, tex=vert_vor,
                              caption='vertex voronoi',
                              cblabel='vertex voronoi')
visb_sc.preview()
