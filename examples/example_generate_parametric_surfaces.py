import trimesh
import slam.generate_parametric_surfaces as sgps
import slam.plot as splt
import numpy as np

if __name__ == '__main__':
    # Quadric
    Ks = [[1, 1]]
    X, Y, faces, Zs = sgps.generate_quadric(Ks, nstep=20)
    Z = Zs[0]

    coords = np.array([X, Y, Z]).transpose()
    quadric = trimesh.Trimesh(faces=faces, vertices=coords, process=False)

    # Ellipsoid Parameters
    nstep = 50
    randomSampling = True
    a = 2
    b = 1
    ellips = sgps.generate_ellipsiod(a, b, nstep, randomSampling)

    # Sphere
    sphere = sgps.generate_sphere(100)
    # compare its volume to the analytical one
    analytical_vol = (4/3)*np.pi
    print(sphere.volume-analytical_vol)

    # plot
    visb_sc = splt.visbrain_plot(mesh=quadric, caption='quadric')
    visb_sc = splt.visbrain_plot(mesh=ellips, caption='ellipsoid',
                                 visb_sc=visb_sc)
    visb_sc = splt.visbrain_plot(mesh=sphere, caption='sphere',
                                 visb_sc=visb_sc)
    visb_sc.preview()
