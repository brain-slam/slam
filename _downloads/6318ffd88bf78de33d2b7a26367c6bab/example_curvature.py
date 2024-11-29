"""
.. _example_curvature:

===================================
example of curvature estimation in slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>
#          Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

# importation of slam modules
import slam.utils as ut
import numpy as np
import slam.generate_parametric_surfaces as sgps
import slam.io as sio
import slam.curvature as scurv

###############################################################################
# loading an examplar mesh
mesh_file = "../examples/data/example_mesh.gii"
mesh = sio.load_mesh(mesh_file)

###############################################################################
# Comptue estimations of principal curvatures
PrincipalCurvatures, PrincipalDir1, PrincipalDir2 \
    = scurv.curvatures_and_derivatives(mesh)

###############################################################################
# Comptue Gauss curvature from principal curvatures
gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]

###############################################################################
# Comptue mean curvature from principal curvatures
mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

###############################################################################
# Decomposition of the curvatures into ShapeIndex and Curvedness
# Based on 'Surface shape and curvature scales
#           Jan JKoenderink & Andrea Jvan Doorn'
shapeIndex, curvedness = scurv.decompose_curvature(PrincipalCurvatures)

###############################################################################
# Estimation error on the principal curvature length
K = [1, 0]

quadric = sgps.generate_quadric(
    K,
    nstep=[20, 20],
    ax=3,
    ay=3,
    random_sampling=False,
    ratio=0.3,
    random_distribution_type="gamma",
    equilateral=True,
)

###############################################################################
# Estimated computation of the Principal curvature, K_gauss, K_mean
p_curv, d1_estim, d2_estim = scurv.curvatures_and_derivatives(quadric)

k1_estim, k2_estim = p_curv[0, :], p_curv[1, :]

k_gauss_estim = k1_estim * k2_estim

k_mean_estim = 0.5 * (k1_estim + k2_estim)

###############################################################################
# Analytical computation of the curvatures

k_mean_analytic = sgps.quadric_curv_mean(K)(
    np.array(quadric.vertices[:, 0]), np.array(quadric.vertices[:, 1])
)

k_gauss_analytic = sgps.quadric_curv_gauss(K)(
    np.array(quadric.vertices[:, 0]), np.array(quadric.vertices[:, 1])
)

k1_analytic = np.zeros((len(k_mean_analytic)))
k2_analytic = np.zeros((len(k_mean_analytic)))

for i in range(len(k_mean_analytic)):
    a, b = np.roots((1, -2 * k_mean_analytic[i], k_gauss_analytic[i]))
    k1_analytic[i] = min(a, b)
    k2_analytic[i] = max(a, b)


###############################################################################
# Error computation

k_mean_relative_change = abs(
    (k_mean_analytic - k_mean_estim) / k_mean_analytic)
k_mean_absolute_change = abs((k_mean_analytic - k_mean_estim))

k1_relative_change = abs((k1_analytic - k1_estim) / k1_analytic)
k1_absolute_change = abs((k1_analytic - k1_estim))

###############################################################################
# Estimation error on the curvature directions
# commented because there is a bug:
# ValueError: shapes (3,2) and (3,2) not aligned: 2 (dim 1) != 3 (dim 0)
# actually, vec1.shape=(3,) while vec2.shape=(3,2)

K = [1, 0]

quadric = sgps.generate_quadric(
    K,
    nstep=[20, 20],
    ax=3,
    ay=3,
    random_sampling=False,
    ratio=0.3,
    random_distribution_type="gamma",
    equilateral=True,
)

###############################################################################
# Estimated computation of the Principal curvature, Direction1, Direction2
p_curv_estim, d1_estim, d2_estim = scurv.curvatures_and_derivatives(quadric)

###############################################################################
# Analytical computation of the directions
analytical_directions = sgps.compute_all_principal_directions_3D(
    K, quadric.vertices)

estimated_directions = np.zeros(analytical_directions.shape)
estimated_directions[:, :, 0] = d1_estim
estimated_directions[:, :, 1] = d2_estim

angular_error_0, dotprods = ut.compare_analytic_estimated_directions(
    analytical_directions[:, :, 0], estimated_directions[:, :, 0]
)
angular_error_0 = 180 * angular_error_0 / np.pi

angular_error_1, dotprods = ut.compare_analytic_estimated_directions(
    analytical_directions[:, :, 1], estimated_directions[:, :, 1]
)
angular_error_1 = 180 * angular_error_1 / np.pi

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# import visbrain # visu using visbrain
# Plot mean curvature
# visb_sc = splt.visbrain_plot(
#     mesh=mesh,
#     tex=mean_curv,
#     caption="mean curvature",
#     cblabel="mean curvature"
# )
# visb_sc.preview()
#############################################################################
# # Plot Gauss curvature
# visb_sc = splt.visbrain_plot(
#     mesh=mesh,
#     tex=gaussian_curv,
#     caption="Gaussian curvature",
#     cblabel="Gaussian curvature",
#     cmap="hot",
# )
# visb_sc.preview()
###############################################################################
# Plot of ShapeIndex and Curvedness
# visb_sc = splt.visbrain_plot(
#     mesh=mesh,
#     tex=shapeIndex,
#     caption="ShapeIndex",
#     cblabel="ShapeIndex",
#     cmap="coolwarm",
# )
# visb_sc.preview()
#
# visb_sc = splt.visbrain_plot(
#     mesh=mesh,
#     tex=curvedness,
#     caption="Curvedness",
#     cblabel="Curvedness",
#     cmap="hot"
# )
# visb_sc.preview()
###############################################################################
# Error plot
# visb_sc = splt.visbrain_plot(
#     mesh=quadric,
#     tex=k_mean_absolute_change,
#     caption="K_mean absolute error",
#     cblabel="K_mean absolute error",
# )
# visb_sc.preview()
# ###############################################################################
# # Error plot
# visb_sc = splt.visbrain_plot(
#     mesh=quadric,
#     tex=angular_error_0,
#     caption="Angular error 0",
#     cblabel="Angular error 0",
# )
# visb_sc.preview()
#
# visb_sc = splt.visbrain_plot(
#     mesh=quadric,
#     tex=angular_error_1,
#     caption="Angular error 1",
#     cblabel="Angular error 1",
# )
# visb_sc.preview()
