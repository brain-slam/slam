import slam.generate_parametric_surfaces as sgps
import slam.plot as splt
import numpy as np

if __name__ == '__main__':
    # Quadric
    K = [1, 1]
    quadric = sgps.generate_quadric(K, nstep=[40, 40], ax=3, ay=1,
                                    random_sampling=False,
                                    ratio=0.3,
                                    random_distribution_type='gamma')

    quadric.show()
    quadric2 = \
        sgps.generate_paraboloid_regular(A=1, nstep=40, ax=3, ay=1,
                                         random_sampling=True,
                                         ratio=0.1,
                                         random_distribution_type='gamma')
    quadric2.show()
    quadric_mean_curv = \
        sgps.quadric_curv_mean(K)(np.array(quadric.vertices[:, 0]),
                                  np.array(quadric.vertices[:, 1]))
    # print(np.min(quadric.vertices, 0))
    # print(np.max(quadric.vertices, 0))

    # Ellipsoid Parameters
    nstep = 50
    randomSampling = True
    a = 2
    b = 1
    ellips = sgps.generate_ellipsiod(a, b, nstep, randomSampling)

    # Sphere with regular sampling
    sphere_regular = sgps.generate_sphere_icosahedron(subdivisions=3, radius=4)
    # Sphere random sampling with the same number of vertices
    sphere_random = \
        sgps.generate_sphere_random_sampling(
            vertex_number=sphere_regular.vertices.shape[0],
            radius=4)
    # compare their volume to the analytical one
    analytical_vol = (4 / 3) * np.pi * np.power(4, 3)
    print('volume error for regular sampling: {:.3f}'.format(
        sphere_regular.volume - analytical_vol))
    print('volume error for random sampling: {:.3f}'.format(
        sphere_random.volume - analytical_vol))

    # plot
    visb_sc = splt.visbrain_plot(mesh=quadric, tex=quadric_mean_curv,
                                 caption='quadric',
                                 cblabel='mean curvature')
    visb_sc = splt.visbrain_plot(mesh=ellips, caption='ellipsoid',
                                 visb_sc=visb_sc)
    visb_sc = splt.visbrain_plot(mesh=sphere_random, caption='sphere_random',
                                 visb_sc=visb_sc)
    visb_sc = splt.visbrain_plot(mesh=sphere_regular, caption='sphere_regular',
                                 visb_sc=visb_sc)
    visb_sc.preview()
