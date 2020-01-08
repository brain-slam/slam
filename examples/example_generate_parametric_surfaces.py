import trimesh
import slam.generate_parametric_surfaces as sgps
import slam.plot as splt
import numpy as np

if __name__ == '__main__':
    # Quadric
    K = [1, 1]
    quadric = sgps.generate_quadric(K, nstep=20)

    # Ellipsoid Parameters
    nstep = 50
    randomSampling = True
    a = 2
    b = 1
    ellips = sgps.generate_ellipsiod(a, b, nstep, randomSampling)

    # Sphere random
    sphere_random = sgps.generate_sphere_random_sampling(vertex_number=100, radius=5)
    # compare its volume to the analytical one
    analytical_vol = (4/3)*np.pi*np.power(5, 3)
    print(sphere_random.volume-analytical_vol)
    # Sphere regular
    sphere_regular = sgps.generate_sphere_icosahedron(subdivisions=3, radius=4)
    analytical_vol = (4/3)*np.pi*np.power(4, 3)
    print(sphere_regular.volume-analytical_vol)

    # plot
    visb_sc = splt.visbrain_plot(mesh=quadric, caption='quadric')
    visb_sc = splt.visbrain_plot(mesh=ellips, caption='ellipsoid',
                                 visb_sc=visb_sc)
    visb_sc = splt.visbrain_plot(mesh=sphere_random, caption='sphere_random',
                                 visb_sc=visb_sc)
    visb_sc = splt.visbrain_plot(mesh=sphere_regular, caption='sphere_regular',
                                 visb_sc=visb_sc)
    visb_sc.preview()
