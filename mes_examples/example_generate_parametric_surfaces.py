"""
.. _example_generation_parametric_surfaces:

===================================
Generating parametric surfaces in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Importation of slam modules
import slam.generate_parametric_surfaces as sgps
import slam.plot as splt
import numpy as np

###############################################################################
# Generating a quadrix surface

K = [1, 1]
quadric = sgps.generate_quadric(K, nstep=20, ax=3, ay=1,
                                random_sampling=True,
                                ratio=0.3,
                                random_distribution_type='gamma')
quadric_mean_curv = \
    sgps.quadric_curv_mean(K)(np.array(quadric.vertices[:, 0]),
                              np.array(quadric.vertices[:, 1]))

visb_sc = splt.visbrain_plot(mesh=quadric, tex=quadric_mean_curv,
                             caption='quadric',
                             cblabel='mean curvature')

visb_sc.preview()


###############################################################################
# Generating an ellipsiods

nstep = 50
randomSampling = True
a = 2
b = 1
ellips = sgps.generate_ellipsiod(a, b, nstep, randomSampling)

visb_sc = splt.visbrain_plot(mesh=ellips, caption='ellipsoid')

visb_sc.preview()


###############################################################################
# Generating a sphere
sphere_regular = sgps.generate_sphere_icosahedron(subdivisions=3, radius=4)


visb_sc = splt.visbrain_plot(mesh=sphere_regular, caption='sphere_regular')

visb_sc.preview()


###############################################################################
# Generating a more randomized sphere (random sampling with the same
# number of vertices)

sphere_random = \
    sgps.generate_sphere_random_sampling(
        vertex_number=sphere_regular.vertices.shape[0],
        radius=4)

visb_sc = splt.visbrain_plot(mesh=sphere_random, caption='sphere_random',
                             visb_sc=visb_sc)

visb_sc.preview()


###############################################################################
# Computation of the volume and volume error of the spheres

analytical_vol = (4 / 3) * np.pi * np.power(4, 3)
print('volume error for regular sampling: {:.3f}'.format(
    sphere_regular.volume - analytical_vol))
print('volume error for random sampling: {:.3f}'.format(
    sphere_random.volume - analytical_vol))
