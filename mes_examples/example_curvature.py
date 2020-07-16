"""
.. _example_curvature:

===================================
example of curvature estimation in slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>
#          Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# importation of slam modules
import slam.io as sio
import slam.plot as splt
import slam.curvature as scurv

###############################################################################
# loading an examplar mesh
mesh_file = '../examples/data/example_mesh.gii'
mesh = sio.load_mesh(mesh_file)

###############################################################################
# Comptue estimations of principal curvatures
PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
    scurv.curvatures_and_derivatives(mesh)

###############################################################################
# Comptue Gauss curvature from principal curvatures
gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]

###############################################################################
# Comptue mean curvature from principal curvatures
mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

###############################################################################
# Plot mean curvature
visb_sc = splt.visbrain_plot(mesh=mesh, tex=mean_curv,
                             caption='mean curvature',
                             cblabel='mean curvature')
visb_sc.preview()

###############################################################################
# Plot Gauss curvature
visb_sc = splt.visbrain_plot(mesh=mesh, tex=gaussian_curv,
                             caption='Gaussian curvature',
                             cblabel='Gaussian curvature',
                             cmap='hot')
visb_sc.preview()

###############################################################################
# Decomposition of the curvatures into ShapeIndex and Curvedness
# Based on 'Surface shape and curvature scales
#           Jan JKoenderink & Andrea Jvan Doorn'
shapeIndex, curvedness = scurv.decompose_curvature(PrincipalCurvatures)

###############################################################################
# Plot of ShapeIndex and Curvedness
visb_sc = splt.visbrain_plot(mesh=mesh, tex=shapeIndex,
                             caption='ShapeIndex',
                             cblabel='ShapeIndex',
                             cmap='hot')

visb_sc = splt.visbrain_plot(mesh=mesh, tex=curvedness,
                             caption='Curvedness',
                             cblabel='Curvedness',
                             cmap='hot', visb_sc=visb_sc)
visb_sc.preview()
