import numpy as np
from scipy.spatial import Delaunay
import trimesh
from trimesh import creation as tcr


def quadric(K1, K2):
    """
    compute the Z coordinate of a quadric dependeing on X and Y coordinates
    :param K1:
    :param K2:
    :return:
    """

    def fonction(x, y):
        return K1 * x ** 2 + K2 * y ** 2

    return fonction


def quadric_curv_gauss(K):
    """
    analytical Gaussian curvature of a quadric
    :param K1:
    :param K2:
    :return:
    """
    K1 = K[0]
    K2 = K[1]

    def curv_gauss(x, y):
        num = -4 * (K1 * K2)
        denom = (1 + 4 * K1 ** 2 * x ** 2 + 4 * K2 ** 2 * y ** 2) ** 2
        return num / denom

    return curv_gauss


def quadric_curv_mean(K):
    """
    analytical mean curvature of a quadric
    :param K1:
    :param K2:
    :return:
    """
    K1 = K[0]
    K2 = K[1]

    def curv_mean(x, y):
        num = -(2 * K2 * (1 + 4 * K1 ** 2 * x ** 2) + 2 * K1 *
                (1 + 4 * K2 ** 2 * y ** 2))
        denom = 2 * (1 + 4 * K1 ** 2 * x ** 2 +
                     4 * K2 ** 2 * y ** 2) ** (3 / 2)

        return num / denom

    return curv_mean


def generate_quadric(K, nstep=50, ax=1, ay=1, random_sampling=True,
                     ratio=0.2, random_distribution_type='gaussian'):
    """
    generate a quadric mesh
    ratio and random_distribution_type parameters are unused if
    random_sampling is set to False
    :param K:
    :param nstep:
    :param ax:
    :param ay:
    :param random_sampling:
    :param ratio:
    :param random_distribution_type:
    :return:
    """

    # Parameters
    xmin, xmax = [-ax, ax]
    ymin, ymax = [-ay, ay]

    # Coordinates
    stepx = (xmax - xmin) / nstep
    x = np.arange(xmin, xmax, stepx)
    # x, stepx = np.linspace(xmin, xmax, nstep, retstep=True)
    stepy = stepx * np.sqrt(3) / 2  # to ensure equilateral faces
    y = np.arange(ymin, ymax, stepy)
    # y = np.linspace(ymin, ymax, nstep)
    X, Y = np.meshgrid(x, y)
    X[::2] += stepx / 2
    # Y += np.sqrt(3) / 2
    X = X.flatten()
    Y = Y.flatten()

    if random_sampling:
        sigma = stepx * ratio  # characteristic size of the mesh * ratio
        nb_vert = len(x) * len(y)
        if random_distribution_type == 'gamma':
            theta = np.random.rand(nb_vert, ) * np.pi * 2
            mean = sigma
            variance = sigma ** 2
            radius = \
                np.random.gamma(mean ** 2 / variance, variance / mean, nb_vert)
            X = X + radius * np.cos(theta)
            Y = Y + radius * np.sin(theta)
        elif random_distribution_type == 'uniform':
            X = X + np.random.uniform(-1, 1, 100)
            Y = Y + np.random.uniform(-1, 1, 100)
        else:
            X = X + sigma * np.random.randn(nb_vert, )
            Y = Y + sigma * np.random.randn(nb_vert, )

    # Delaunay triangulation, based on scipy binding of Qhull.
    # See https://scipy.github.io/devdocs/generated/scipy.spatial.Delaunay.html#scipy.spatial.Delaunay
    # and http://www.qhull.org/html/qdelaun.htm for more informations
    faces_tri = Delaunay(np.vstack((X, Y)).T, qhull_options='Qj Qt Qbb')# Qbb Qc Qz Qj')

    Z = quadric(K[0], K[1])(X, Y)
    coords = np.array([X, Y, Z]).transpose()

    return trimesh.Trimesh(faces=faces_tri.simplices,
                           vertices=coords,
                           process=False)


def generate_ellipsiod(a, b, nstep, random_sampling=False):
    """
    generate an ellipsoid
    :param a:
    :param b:
    :param nstep:
    :param random_sampling:
    :return:
    """
    # Coordinates
    if random_sampling:
        THETA = (np.random.rand(nstep * nstep, 1) - 1 / 2) * np.pi
        PHI = 2 * np.pi * np.random.rand(nstep * nstep, 1)
    else:
        theta = np.linspace(-np.pi / 2, np.pi / 2, nstep)
        phi = np.linspace(0, 2 * np.pi, nstep)
        THETA, PHI = np.meshgrid(theta, phi)

    # Sphere coordinates
    X = a * np.cos(THETA) * np.cos(PHI)
    Y = b * np.cos(THETA) * np.sin(PHI)
    Z = np.sin(THETA)

    coords = np.array([X, Y, Z]).squeeze().transpose()

    return tri_from_hull(coords)


def tri_from_hull(vertices):
    """
    compute faces from vertices using trimesh convex hull
    :param vertices: (n, 3) float
    :return:
    """
    mesh = trimesh.Trimesh(vertices=vertices, process=False)
    return mesh.convex_hull


def generate_sphere_random_sampling(vertex_number=100, radius=1.0):
    """
    generate a sphere with random sampling
    :param vertex_number: number of vertices in the output spherical mesh
    :param radius: radius of the output sphere
    :return:
    """
    coords = np.zeros((vertex_number, 3))
    for i in range(vertex_number):
        M = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
        coords[i, :] = Q[:, 0].transpose() * np.sign(R[0, 0])
    if radius != 1:
        coords = radius * coords
    return tri_from_hull(coords)


def generate_sphere_icosahedron(subdivisions=3, radius=1.0):
    """
    generate a sphere by subdividing an icosahedron
    simply call the trimesh function
    see trimesh.creation.icosphere for more details
    :param subdivisions:  int
      How many times to subdivide the mesh.
      Note that the number of faces will grow as function of
      4 ** subdivisions, so you probably want to keep this under ~5
    :param radius: float
      Desired radius of sphere
    :return:
    """
    return tcr.icosphere(subdivisions=subdivisions, radius=radius)
