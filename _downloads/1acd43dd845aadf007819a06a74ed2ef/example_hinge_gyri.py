"""
.. _example_curvature:

===================================
example of hinge shaped surface
===================================
"""

# Authors: Julien Lefevre <julien.lefevre@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# importation of slam modules
import slam.curvature as scurv
import slam.generate_parametric_surfaces as sgps
import numpy as np

###############################################################################
# Creating an examplar 3-4-...n hinge mesh
hinge_mesh = sgps.generate_hinge(n_hinge=4)
mesh_curvatures = scurv.curvatures_and_derivatives(hinge_mesh)
mean_curvature = 1 / 2 * mesh_curvatures[0].sum(axis=0)

#############################################################################
# VISUALIZATION USING plotly
# #############################################################################

import slam.plot as splt

display_settings = {}
display_settings['colorbar_label'] = 'Mean Curvature'
mesh_data = {}
mesh_data['vertices'] = hinge_mesh.vertices
mesh_data['faces'] = hinge_mesh.faces
mesh_data['title'] = 'Hinge'
intensity_data = {}
intensity_data['values'] = mean_curvature
intensity_data["mode"] = "vertex"
fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
fig.show()
fig

