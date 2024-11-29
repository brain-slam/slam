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

###############################################################################
# Creating an examplar 3-4-...n hinge mesh
hinge_mesh = sgps.generate_hinge(n_hinge=4)
mesh_curvatures = scurv.curvatures_and_derivatives(hinge_mesh)
mean_curvature = 1 / 2 * mesh_curvatures[0].sum(axis=0)

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
# #############################################################################
# # Visualization with visbrain
# visb_sc = splt.visbrain_plot(
#     mesh=hinge_mesh,
#     tex=mean_curvature,
#     caption="hinge",
#     cblabel="mean curvature"
# )
# visb_sc.preview()
