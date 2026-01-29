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
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# import slam.plot as splt
# ###############################################################################
# # Visualization
# visb_sc = splt.visbrain_plot(
#     mesh=mesh, tex=vert_vor,
#     caption="vertex voronoi",
#     cblabel="vertex voronoi"
# )
# visb_sc.preview()
