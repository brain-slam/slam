"""
.. _example_generation_parametric_surfaces:

===================================
Generating parametric surfaces in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# Importation of slam modules
import slam.generate_parametric_surfaces as sgps
import numpy as np

###############################################################################
# Generating a quadrix surface
K = [1, 1]
quadric = sgps.generate_quadric(
    K,
    nstep=[20, 20],
    ax=3,
    ay=1,
    random_sampling=True,
    ratio=0.3,
    random_distribution_type="gamma",
)
quadric_mean_curv = sgps.quadric_curv_mean(K)(
    np.array(quadric.vertices[:, 0]), np.array(quadric.vertices[:, 1])
)

###############################################################################
# Generating an ellipsiods
nstep = 50
randomSampling = True
a = 2
b = 1
ellips = sgps.generate_ellipsiod(a, b, nstep, randomSampling)

###############################################################################
# Generating a sphere
sphere_regular = sgps.generate_sphere_icosahedron(subdivisions=3, radius=4)

###############################################################################
# Generating a more randomized sphere (random sampling with the same
# number of vertices)
sphere_random = sgps.generate_sphere_random_sampling(
    vertex_number=sphere_regular.vertices.shape[0], radius=4
)

###############################################################################
# Computation of the volume and volume error of the spheres
analytical_vol = (4 / 3) * np.pi * np.power(4, 3)
print(
    "volume error for regular sampling: {:.3f}".format(
        sphere_regular.volume - analytical_vol
    )
)
print(
    "volume error for random sampling: {:.3f}".format(
        sphere_random.volume - analytical_vol
    )
)

#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################

import slam.plot as splt

# Plot Mean Curvature
display_settings = {}
display_settings['colorbar_label'] = 'Curvature'
mesh_data = {}
mesh_data['vertices'] = quadric.vertices
mesh_data['faces'] = quadric.faces
mesh_data['title'] = 'Mean Curvature'
intensity_data = {}
intensity_data['values'] = quadric_mean_curv
intensity_data["mode"] = "vertex"
Fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

# show the ellipsoid
display_settings = {}
intensity_data = None
mesh_data = {}
mesh_data['vertices'] = ellips.vertices
mesh_data['faces'] = ellips.faces
mesh_data['title'] = 'Ellipsoid Mesh'
Fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

# show the sphere with regular sampling
display_settings = {}
intensity_data = None
display_settings['colorbar_label'] = 'Curvature'
mesh_data = {}
mesh_data['vertices'] = sphere_regular.vertices
mesh_data['faces'] = sphere_regular.faces
mesh_data['title'] = 'Sphere Regular Mesh'
Fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

# # show the sphere with regular sampling
display_settings = {}
intensity_data = None
mesh_data = {}
mesh_data['vertices'] = sphere_random.vertices
mesh_data['faces'] = sphere_random.faces
mesh_data['title'] = 'Sphere Random Mesh'
Fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()
