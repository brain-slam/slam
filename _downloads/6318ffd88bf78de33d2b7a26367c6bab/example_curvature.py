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
import slam.plot as splt

vertices = mesh.vertices
# center the vertices
vertices = vertices - np.mean(vertices, axis=0)
vertices_translate = np.copy(vertices)
# rotate the vertices
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, vertices_translate.T).T
rot_z = np.array([[np.cos(theta), -np.sin(theta),0],
                  [np.sin(theta),  np.cos(theta),0],
                  [0, 0, 1],])
vertices_translate = np.dot(rot_z, vertices_translate.T).T
display_settings = {}
display_settings['colorbar_label'] = 'Mean Curvature'
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'example_mesh.gii Mean Curvature'
intensity_data = {}
intensity_data['values'] = mean_curv
intensity_data["mode"] = "vertex"
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
#Fig.show()
Fig.write_image("example_curvature_1.png")

mesh_data['title'] = 'example_mesh.gii Gaussian Curvature'
intensity_data['values'] = gaussian_curv
display_settings['colorbar_label'] = 'Gaussian Curvature'
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
#Fig.show()
Fig.write_image("example_curvature_2.png")

mesh_data['title'] = 'example_mesh.gii Shape Index'
intensity_data['values'] = shapeIndex
display_settings['colorbar_label'] = 'Shape Index'
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
#Fig.show()
Fig.write_image("example_curvature_3.png")

mesh_data['title'] = 'example_mesh.gii Curvedness'
intensity_data['values'] = curvedness
display_settings['colorbar_label'] = 'Curvedness'
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
#Fig.show()
Fig.write_image("example_curvature_4.png")

mesh_data['vertices'] = quadric.vertices
mesh_data['faces'] = quadric.faces
mesh_data['title'] = 'Quadric K Mean Absolute Change'
intensity_data['values'] = k_mean_absolute_change
display_settings['colorbar_label'] = 'K Mean Absolute Change'
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
#Fig.show()
Fig.write_image("example_curvature_5.png")

mesh_data['title'] = 'Quadric Angular Error 0'
intensity_data['values'] = angular_error_0
display_settings['colorbar_label'] = 'Angular Error 0'
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.write_image("example_curvature_6.png")

mesh_data['title'] = ('Quadric Angular Error 1')
intensity_data['values'] = angular_error_1
display_settings['colorbar_label'] = 'Angular Error 1'
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
#Fig.show()
Fig.write_image("example_curvature_7.png")
