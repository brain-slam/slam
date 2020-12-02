import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import newton
import trimesh
from trimesh import creation as tcr
import slam.topology as stop


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
        num = 4 * (K1 * K2)
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


def adaptive_sampling(ymax, K, step):
    """
        sample [-ymax,ymax] such as:
        1. y_{i+1}-y_i =h_i with h_i obtained by a recursive formula (1)
        2. f(y_i) is regularly sampled, where f(x)=K x**2

        => same as
        1. Computing the curvilinear abscissa of x -> Kx**2 and
        sample regularly with parameter step *sqrt(3)/2
        s(x) = [ 2*K*x*sqrt((2*K*x)**2+1) + arcsinh(2*K*x) ]/ (4*K)
        2. Come back in the y domain by inverting the curvilinear abscissa
        (Newton method)

    :param ymax:
    :param K: amplitude of the paraboloid
    :param step: desired sampling step if K =0
    :return:
    """
    # Curvilinear abscisse
    def f(x):
        return (2 * K * x * np.sqrt((2 * K * x) ** 2 + 1) +
                np.arcsinh(2 * K * x)) / (4 * K)

    # Step 1
    curve_length = f(ymax)
    curve_step = np.sqrt(3) / 2 * step  # * np.sqrt(K+1) # Pythagore
    Npoints = int(np.floor(curve_length / curve_step))
    curve_samples = np.arange(0, curve_length, curve_step)

    # Step 2
    y_pos = np.zeros((Npoints + 1,))
    for i in range(Npoints + 1):
        y_pos[i] = newton(lambda x:
                          f(x) - curve_samples[i], curve_samples[i])
    y_pos = np.concatenate([-y_pos[::-1], y_pos[1:]])
    curve_samples = np.concatenate([-curve_samples[::-1], curve_samples[1:]])
    return y_pos, curve_samples


def generate_paraboloid_regular(A, nstep=50, ax=1, ay=1,
                                random_sampling=False,
                                random_distribution_type='gaussian',
                                ratio=0.1):
    """
        generate a regular paraboloid mesh Z=K*Y^2
        ratio and random_distribution_type parameters are unused if
        random_sampling is set to False
        :param A: amplitude of the paraboloid
        :param nstep: nstepx or the sampling step stepx as a float !
        :param ax: half length of the domain
        :param ay: half width of the domain
        :param random_sampling:
        :param random_distribution_type:
        :param ratio:
        :return:
        """
    # Parameters
    xmin, xmax = [-ax, ax]
    ymax = ay    # ymin, ymax = [-ay, ay]
    # Define the sampling
    if isinstance(nstep, int):
        stepx = (xmax - xmin) / nstep
    else:
        stepx = nstep

    # Coordinates
    x = np.arange(xmin, xmax, stepx)
    # To generate y
    y, curve_samples = adaptive_sampling(ymax, A, stepx)

    X, Y = np.meshgrid(x, y)
    X[::2] += stepx / 2
    X = X.flatten()
    Y = Y.flatten()

    # Random perturbation
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

    # Delaunay triangulation: be careful, to do on the curvilinear aspects to
    # avoid triangle flips
    Xtmp, S = np.meshgrid(x, curve_samples)
    S = S.flatten()
    # faces_tri = Triangulation(X, S)
    faces_tri = Delaunay(np.vstack((X, S)).T, qhull_options='QJ Qt Qbb')
    # alternative setting? 'Qbb Qc Qz Qj'

    Z = quadric(0, A)(X, Y)
    coords = np.array([X, Y, Z]).transpose()

    paraboloid_mesh = trimesh.Trimesh(faces=faces_tri.simplices,
                                      vertices=coords,
                                      process=False)
    # remove the faces having any vertex on the boundary to avoid
    # atypical faces geometry due to Delaunay triangulation in 2D
    # TO DO: same boundary removal as for generate_quadric
    return stop.remove_mesh_boundary_faces(paraboloid_mesh,
                                           face_vertex_number=1)


"""
older, outdated version
# def generate_quadric(K, nstep=50, ax=1, ay=1, random_sampling=True,
#                      ratio=0.2, random_distribution_type='gaussian'):
#     # Parameters
#     xmin, xmax = [-ax, ax]
#     ymin, ymax = [-ay, ay]
#
#     # Coordinates
#     stepx = (xmax - xmin) / nstep
#     x = np.arange(xmin, xmax, stepx)
#     # x, stepx = np.linspace(xmin, xmax, nstep, retstep=True)
#     stepy = stepx * np.sqrt(3) / 2  # to ensure equilateral faces
#     y = np.arange(ymin, ymax, stepy)
#     # y = np.linspace(ymin, ymax, nstep)
#     X, Y = np.meshgrid(x, y)
#     X[::2] += stepx / 2
#     # Y += np.sqrt(3) / 2
#     X = X.flatten()
#     Y = Y.flatten()
#
#     if random_sampling:
#         sigma = stepx * ratio  # characteristic size of the mesh * ratio
#         nb_vert = len(x) * len(y)
#         if random_distribution_type == 'gamma':
#             theta = np.random.rand(nb_vert, ) * np.pi * 2
#             mean = sigma
#             variance = sigma ** 2
#             radius = \
#               np.random.gamma(mean ** 2 / variance, variance / mean, nb_vert)
#             X = X + radius * np.cos(theta)
#             Y = Y + radius * np.sin(theta)
#         elif random_distribution_type == 'uniform':
#             X = X + np.random.uniform(-1, 1, 100)
#             Y = Y + np.random.uniform(-1, 1, 100)
#         else:
#             X = X + sigma * np.random.randn(nb_vert, )
#             Y = Y + sigma * np.random.randn(nb_vert, )
#
#     faces_tri = Delaunay(np.vstack((X, Y)).T, qhull_options='QJ Qt Qbb')
#     # alternative settings? 'Qbb Qc Qz Qj'
#
#     Z = quadric(K[0], K[1])(X, Y)
#     coords = np.array([X, Y, Z]).transpose()
#
#     quadric_mesh = trimesh.Trimesh(faces=faces_tri.simplices,
#                                    vertices=coords,
#                                    process=False)
#     # remove the faces having any vertex on the boundary to avoid
#     # atypical faces geometry due to Delaunay triangulation in 2D
#    return stop.remove_mesh_boundary_faces(quadric_mesh, face_vertex_number=1)
"""


def generate_quadric(K, nstep=[int(50), int(50)], equilateral=False,
                     ax=1, ay=1, random_sampling=True,
                     ratio=0.2, random_distribution_type='gaussian'):
    """
    generate a quadric mesh Z=K1*X^2 + K2*Y^2
    ratio and random_distribution_type parameters are unused if
    random_sampling is set to False
    :param K: list with [K1,K2]
    :param nstep: list with [nstepx,nstepy] or the sampling steps
    [stepx,stepy] as floats !
    :param equilateral: to have an equilateral sampling scheme of the quadric
    :param ax: half length of the domain
    :param ay: half width of the domain
    :param random_sampling:
    :param ratio:
    :param random_distribution_type:
    :return:
    """

    # Parameters
    xmin, xmax = [-ax, ax]
    ymin, ymax = [-ay, ay]
    # Define the sampling
    if equilateral:
        if isinstance(nstep[0], int):
            stepx = (xmax - xmin) / nstep[0]
        else:
            stepx = nstep[0]
        stepy = stepx * np.sqrt(3) / 2  # to ensure equilateral faces
    else:
        if isinstance(nstep[0], int):
            stepx = (xmax - xmin) / nstep[0]
            stepy = (ymax - ymin) / nstep[1]
        else:
            stepx = nstep[0]
            stepy = nstep[1]

    # Coordinates
    x = np.arange(xmin, xmax, stepx)
    # x, stepx = np.linspace(xmin, xmax, nstep, retstep=True)
    y = np.arange(ymin, ymax, stepy)
    # y = np.linspace(ymin, ymax, nstep)
    X, Y = np.meshgrid(x, y)

    # Boundary of a meshgrid-like set of points,
    # warning X.max()/Y.max() and not xmax, ymax
    boundary_x = np.logical_or(X.flatten() == xmin, X.flatten() == X.max())
    boundary_y = np.logical_or(Y.flatten() == ymin, Y.flatten() == Y.max())
    boundary = np.logical_or(boundary_x, boundary_y)
    boundary = np.where(boundary)[0]

    # Adapt in case of equilateral meshing
    if equilateral:
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
    # See https://scipy.github.io/devdocs/generated/scipy.spatial.Delaunay.html
    #     scipy.spatial.Delaunay
    #     and http://www.qhull.org/html/qdelaun.htm for more informations
    faces_tri = Delaunay(np.vstack((X, Y)).T, qhull_options='QJ Qt Qbb')
    # alternative settings? 'Qbb Qc Qz Qj'

    Z = quadric(K[0], K[1])(X, Y)
    coords = np.array([X, Y, Z]).transpose()

    quadric_mesh = trimesh.Trimesh(faces=faces_tri.simplices, vertices=coords,
                                   process=False)

    # Remove boundary, computed previously
    boundary_faces = stop.ismember(quadric_mesh.faces, boundary)
    # compute the mask of faces to keep
    # faces for which face_vertex_number or more vertices are on the boundary
    # are excluded from the mask.
    # i.e. faces with 3-face_vertex_number vertices on the boundary are kept
    face_mask = np.logical_not(np.sum(boundary_faces, 1) >= 3)
    quadric_mesh.update_faces(face_mask)
    quadric_mesh.remove_unreferenced_vertices()

    return quadric_mesh


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


def compute_weingarten_map(K, x, y):
    """
    compute the weingarten matrix of a quadric depending on x and y coordinates
    :param K: (2,) array such as z(x,y)=K[0]*x**2 + K[1]*y**2
    :param (x,y): coordinates where to compute the matrix
    :return: (2,2) matrix of the weingarten endomorphism to be evaluated at
    (x,y)
    """
    K1 = K[0]
    K2 = K[1]
    M = np.zeros((2, 2), dtype=float)
    coeff = 4 * K1 * K2 * x * y
    M[0, 0] = K1 * (1 + 4 * K2 ** 2 * y ** 2)
    M[0, 1] = -coeff * K2
    M[1, 0] = -coeff * K1
    M[1, 1] = K2 * (1 + 4 * K1 ** 2 * x ** 2)
    return M


def compute_principal_directions(K, x, y):
    """
    Compute the principal direction of a quadric, obtained as eigenvectors of
    the Weingarten matrix
    :param K: (2,) array such as z(x,y)=K[0]*x**2 + K[1]*y**2
    :param x: x coordinate where to compute the matrix
    :param y: y coordinate where to compute the matrix
    :return: a 2-list of (2,1) vectors, i.e. principal directions ordered with
    Kmin,Kmax respectively
    """
    M = compute_weingarten_map(K, x, y)
    w, v = np.linalg.eig(M)
    if w[0] > w[1]:
        return v[:, 1], v[:, 0]
    return v[:, 0], v[:, 1]


def compute_local_basis(K, x, y):
    """
    Compute the local basis of the tangent plane for the quadric
    z(x,y)=K[0]*x**2+K[1]*y**2
    :param K: (2,) array such as z(x,y)=K[0]*x**2 + K[1]*y**2
    :param x: x coordinate where to compute the matrix
    :param y: y coordinate where to compute the matrix
    :return: a 2-list of (3,1) vectors, obtained as
    d (x,y,z(x,y) /dx and d (x,y,z(x,y) /dy
    """
    e1 = np.array([[1, 0, 2 * K[0] * x]])
    e2 = np.array([[0, 1, 2 * K[1] * y]])
    return e1, e2


def compute_all_principal_directions(K, vertices):
    """
    Compute all the principal directions of a quadric surface defined through
    the input K and sampled points vertices in a local basis
    :param K: (2,) array such as z(x,y)=K[0]*x**2 + K[1]*y**2
    :param vertices: (n,3) coordinates of the quadric. It is implicitly assumed
    that vertices[:,0] and vertices[:,1]
    corresponds to x and y axis
    :return: res: a (n,2,2) array such as res[i,:,0] and res[i,:,1] are the two
    principal directions at vertex vertices[i,:] in the local basis
    """
    n = vertices.shape[0]
    res = np.zeros((n, 2, 2), dtype=float)
    for i in range(n):
        u1, u2 = compute_principal_directions(K, vertices[i, 0],
                                              vertices[i, 1])
        res[i, :, 0] = u1
        res[i, :, 1] = u2
    return res


def compute_all_principal_directions_3D(K, vertices):
    """
    Compute all the principal directions of a quadric surface defined through
    the input K and sampled points vertices in 3D
    :param K: (2,) array such as z(x,y)=K[0]*x**2 + K[1]*y**2
    :param vertices: (n,3) coordinates of the quadric. It is implicitly assumed
    that vertices[:,0] and vertices[:,1]
    corresponds to x and y axis
    :return: res: a (n,3,2) array such as res[i,:,0] and res[i,:,1] are the two
    principal directions at vertex vertices[i,:]
    """
    n = vertices.shape[0]
    res = np.zeros((n, 3, 2), dtype=float)
    for i in range(n):
        u1, u2 = compute_principal_directions(K, vertices[i, 0],
                                              vertices[i, 1])
        e1, e2 = compute_local_basis(K, vertices[i, 0], vertices[i, 1])
        res[i, :, 0] = u1[0] * e1 + u1[1] * e2
        res[i, :, 1] = u2[0] * e1 + u2[1] * e2
    return res
