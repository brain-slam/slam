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
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

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
# VISUALIZATION USING INTERNAL TOOLS
# #############################################################################

import slam.plot as splt

hinge_mesh.apply_transform(hinge_mesh.principal_inertia_transform)
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, hinge_mesh.vertices.T).T

display_settings = {}
display_settings['colorbar_label'] = 'Mean Curvature'
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = hinge_mesh.faces
mesh_data['title'] = 'Hinge'
intensity_data = {}
intensity_data['values'] = mean_curvature
intensity_data["mode"] = "vertex"
Fig = splt.mesh_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
# Fig.show()
Fig.write_image("example_hinge_gyri_1.png")
