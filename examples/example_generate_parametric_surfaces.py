import trimesh
import slam.generate_parametric_surfaces as sps
import numpy as np

if __name__ == '__main__':

    Ks = [[1, 1]]
    X, Y, faces, Zs = sps.generate_quadric(Ks, nstep=10)
    Z = Zs[0]

    coords = np.array([X, Y, Z]).transpose()
    mesh = trimesh.Trimesh(faces=faces, vertices=coords, process=False)
    mesh.show()

    # Ellipsoid Parameters
    nstep = 50
    randomSampling = True
    a = 2
    b = 1
    ellips = sps.generate_ellipsiod(a, b, nstep, randomSampling)
    ellips.show()

    sphere = sps.generate_sphere(10)
    sphere.show()
