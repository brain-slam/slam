import numpy as np
from trimesh import util as tut

# from trimesh.geometry import mean_vertex_normals
import slam.topology as stop


def norm(vector):
    return np.sqrt(np.sum(vector ** 2))


def curvature_fit(mesh, tol=1e-12, neighbour_size=2):
    """
    Computation of the two principal curvatures based on:
    Petitjean, A survey of methods for recovering quadrics
    in triangle meshes, ACM Computing Surveys, 2002
    :param mesh:
    :param tol:
    :param neighbour_size:
    :return:
    """
    N = mesh.vertices.shape[0]
    # vertex_normals = mean_vertex_normals(N,
    # faces=mesh.faces, face_normals = mesh.face_normals)
    # vertex_normals, Avertex, Acorner, up, vp =
    # calcvertex_normals(mesh, mesh.face_normals)
    vertex_normals = mesh.vertex_normals

    curvature = np.zeros((N, 2))
    directions = np.zeros((N, 3, 2))

    adjacency_matrix = stop.adjacency_matrix(mesh)

    for i in range(N):
        if i == 175:
            print("Debug")
        # Definition of local basis
        point = np.reshape(mesh.vertices[i, :], (3, 1))
        normal = vertex_normals[i, :].transpose()
        normal = normal / norm(normal)
        normal = np.reshape(normal, (3, 1))
        vec1, vec2 = determine_local_basis(normal, tol)
        # rotation_matrix = np.concatenate((vec1.transpose(),
        #                                  np.reshape(vec2, (1, 3)),
        #                                  normal.transpose()))
        rotation_matrix = np.concatenate(
            (vec1, vec2, normal), axis=1).transpose()

        # neighbours
        neigh = stop.k_ring_neighborhood(
            mesh, index=i, k=neighbour_size, adja=adjacency_matrix
        )
        # neigh =
        # np.logical_and(neigh <= neighbour_size, neigh > 0).nonzero()[0]
        neigh = (neigh <= neighbour_size).nonzero()[0]
        neigh_len = len(neigh)
        vertices_neigh = mesh.vertices[neigh, :].transpose()
        # translation, origin at point
        vertices_neigh = vertices_neigh - np.repeat(point, neigh_len, axis=1)
        rotated_vertices_neigh = np.matmul(rotation_matrix, vertices_neigh)

        # Parametric model
        X = np.reshape(rotated_vertices_neigh[0, :], (neigh_len, 1))
        Y = np.reshape(rotated_vertices_neigh[1, :], (neigh_len, 1))
        Z = np.reshape(rotated_vertices_neigh[2, :], (neigh_len, 1))
        XY = np.concatenate((X ** 2, X * Y, Y ** 2), axis=1)
        parameters = np.matmul(np.linalg.pinv(XY), Z)
        # parameters = np.linalg.lstsq(XY,Z)

        # Curvature tensor
        tensor = np.array(
            [
                [parameters[0, 0], parameters[1, 0] / 2],
                [parameters[1, 0] / 2, parameters[2, 0]],
            ]
        )
        eigval, eigvec = np.linalg.eig(tensor)
        curvature[i, 0] = 2 * eigval[0]
        curvature[i, 1] = 2 * eigval[1]
        directions[i, :, 0] = np.matmul(
            rotation_matrix[0:2, :].transpose(), eigvec[:, 0]
        ).transpose()
        directions[i, :, 1] = np.matmul(
            rotation_matrix[0:2, :].transpose(), eigvec[:, 1]
        ).transpose()

    # Sort
    indices = np.argsort(curvature, axis=1)
    curvature = np.sort(curvature, axis=1)
    indices = np.expand_dims(indices, axis=1)
    directions = np.take_along_axis(directions, indices, axis=2)

    return curvature, directions


def determine_local_basis(normal, tol):
    """Compute an orthonormal basis of a 2d plane defined by its normal.

    A test vector (1,0,0) is first projected on the 2d plane.
    The normalized projection vec1 constituting the first vector of the basis.
    The second vector vec2 is obtained using cross product.

    :param normal: unitary normal vector to the plan
    :type normal: (3,1) ndarray
    :param tol: minimal norm value
    :type: np.float64
    :return:  vec1, vec2 two orthonormal vectors generating the plan
    :rtype: (2,) tuple of ndarray each of size (3,1)
    """
    # (3,3) projection matrix on the tangent plane
    proj_matrix = np.identity(3) - np.matmul(normal, normal.transpose())
    # project (1,0,0) orthogonaly on the plane
    vec1 = np.matmul(proj_matrix, np.array([[1], [0], [0]]))
    # vec1 vanishes if the normal is close to (1,0,0)
    if np.abs(norm(vec1)) < tol:
        vec1 = np.matmul(proj_matrix, np.array([[0], [1], [0]]))
    vec1 = vec1 / norm(vec1)
    vec2 = np.cross(normal[:, 0], vec1[:, 0])
    vec2 = vec2 / norm(vec2)
    vec2 = np.reshape(vec2, (3, 1))
    return vec1, vec2


def project_curvature_tensor(uf, vf, nf, old_ku, old_kuv, old_kv, up, vp):
    """
    performs a projection of the tensor variables to the vertex
    coordinate system
    :param uf:face coordinates system
    :param vf:face coordinates system
    :param nf:face normal
    :param old_ku:face curvature tensor variables
    :param old_kuv:face curvature tensor variables
    :param old_kv:face curvature tensor variables
    :param up:vertex cordinate system
    :param vp:vertex cordinate system
    :return:
    new_ku,new_kuv,new_kv : vertex curvature tensor coordinates
    The tensor : [[new_ku, new_kuv], [new_kuv, new_kv]]
    """

    r_new_u, r_new_v = rotate_coordinate_system(up, vp, nf)
    OldTensor = np.array([[old_ku, old_kuv], [old_kuv, old_kv]])
    u1 = np.dot(r_new_u, uf)
    v1 = np.dot(r_new_u, vf)
    u2 = np.dot(r_new_v, uf)
    v2 = np.dot(r_new_v, vf)

    new_ku = np.dot(
        np.array([u1, v1]), np.dot(OldTensor, np.transpose(np.array([u1, v1])))
    )
    new_kuv = np.dot(
        np.array([u1, v1]), np.dot(OldTensor, np.transpose(np.array([u2, v2])))
    )
    new_kv = np.dot(
        np.array([u2, v2]), np.dot(OldTensor, np.transpose(np.array([u2, v2])))
    )
    return new_ku, new_kuv, new_kv


def compute_curvature(FV, VertexNormals, FaceNormals,
                      Avertex, Acorner, up, vp):
    """
    CalcFaceCurvature recives a list of vertices and faces in FV structure
    and the normal at each vertex and calculates the second fundemental
    matrix and the curvature using least squares
    :param FV: face-vertex data structure containing a list of vertices and a
    list of faces
    :param VertexNormals: n*3 matrix ( n = number of vertices ) containing the
    normal at each vertex
    :param FaceNormals: m*3 matrix ( m = number of faces ) containing the
    normal of each face
    :param Avertex:
    :param Acorner:
    :param up:
    :param vp:
    :return:
    FaceSFM - an m*1 cell matrix second fundemental
    VertexSFM - an n*w cell matrix second fundementel
    wfp - corner voronoi weights
    """
    print("Calculating curvature tensors ... Please wait")
    "Matrix of each face at each cell"
    FaceSFM, VertexSFM = list(), list()
    for i in range(FV.faces.shape[0]):
        FaceSFM.append([[0, 0], [0, 0]])
    for i in range(FV.vertices.shape[0]):
        VertexSFM.append([[0, 0], [0, 0]])
    Kn = np.zeros((1, FV.faces.shape[0]))

    " Get all the edge vectors "
    e0 = FV.vertices[FV.faces[:, 2], :] - FV.vertices[FV.faces[:, 1], :]
    e1 = FV.vertices[FV.faces[:, 0], :] - FV.vertices[FV.faces[:, 2], :]
    e2 = FV.vertices[FV.faces[:, 1], :] - FV.vertices[FV.faces[:, 0], :]

    " Normalize edge vectors "
    e0_norm = tut.unitize(e0)
    # e1_norm = normr(e1)
    # e2_norm = normr(e2)

    wfp = np.array(np.zeros((FV.faces.shape[0], 3)))
    for i in range(FV.faces.shape[0]):
        "Calculate Curvature Per Face"
        "set face coordinate frame"
        nf = FaceNormals[i, :]
        t = e0_norm[i, :]
        B = np.cross(nf, t)
        B = B / (np.linalg.norm(B))

        "extract relevant normals in face vertices"
        n0 = VertexNormals[FV.faces[i][0], :]
        n1 = VertexNormals[FV.faces[i][1], :]
        n2 = VertexNormals[FV.faces[i][2], :]

        " solve least squares problem of th form Ax=b "
        A = np.array(
            [
                [np.dot(e0[i, :], t), np.dot(e0[i, :], B), 0],
                [0, np.dot(e0[i, :], t), np.dot(e0[i, :], B)],
                [np.dot(e1[i, :], t), np.dot(e1[i, :], B), 0],
                [0, np.dot(e1[i, :], t), np.dot(e1[i, :], B)],
                [np.dot(e2[i, :], t), np.dot(e2[i, :], B), 0],
                [0, np.dot(e2[i, :], t), np.dot(e2[i, :], B)],
            ]
        )

        b = np.array(
            [
                np.dot(n2 - n1, t),
                np.dot(n2 - n1, B),
                np.dot(n0 - n2, t),
                np.dot(n0 - n2, B),
                np.dot(n1 - n0, t),
                np.dot(n1 - n0, B),
            ]
        )

        "Resolving by least mean square method because "
        "A is not a square matrix"

        x = np.linalg.lstsq(A, b, None)

        FaceSFM[i] = np.array([[x[0][0], x[0][1]], [x[0][1], x[0][2]]])
        Kn[0][i] = np.dot(
            np.array([1, 0]), np.dot(FaceSFM[i], np.array([[1.0], [0.0]]))
        )
        """
        Calculate curvature per vertex
        Calculate voronoi weights
        """
        wfp[i][0] = Acorner[i][0] / Avertex[FV.faces[i][0]]
        wfp[i][1] = Acorner[i][1] / Avertex[FV.faces[i][1]]
        wfp[i][2] = Acorner[i][2] / Avertex[FV.faces[i][2]]

        "Calculate new coordinate system and project the tensor"

        for j in range(3):
            new_ku, new_kuv, new_kv = project_curvature_tensor(
                t,
                B,
                nf,
                x[0][0],
                x[0][1],
                x[0][2],
                up[FV.faces[i][j], :],
                vp[FV.faces[i][j], :],
            )
            VertexSFM[FV.faces[i][j]] += np.dot(
                wfp[i][j], np.array([[new_ku, new_kuv], [new_kuv, new_kv]])
            )

    print("Finished Calculating curvature tensors")

    return FaceSFM, VertexSFM, wfp


def curvatures_and_derivatives(mesh):
    (VertexNormals, Avertex, Acorner, up, vp) = calcvertex_normals(
        mesh, mesh.face_normals
    )
    (FaceSFM, VertexSFM, wfp) = compute_curvature(
        mesh, VertexNormals, mesh.face_normals, Avertex, Acorner, up, vp
    )
    [PrincipalCurvature, PrincipalDi1, PrincipalDi2] = principal_curvatures(
        mesh, VertexSFM, up, vp
    )
    return PrincipalCurvature, PrincipalDi1, PrincipalDi2


# def calcfacenormals(FV):
#     """
#     Calculates face normals
#     :param FV: face vertex data structure containing a list of vertices and a
#     list of faces
#     :return:
#     Face normals : matrix that contains each faces' normal
#     (dim = number of faces * 3 )
#     """
#
#     "Get all edge vectors"
#     e0 = FV.vertices[FV.faces[:, 2], :] - FV.vertices[FV.faces[:, 1], :]
#     e1 = FV.vertices[FV.faces[:, 0], :] - FV.vertices[FV.faces[:, 2], :]
#
#     "Calculate and return normal of face"
#     return normr(np.cross(e0, e1))


# def normr(X):
#     """
#     Returns a matrix with the same size where each row normalized to a
#     vector length of 1
#     :param X:
#     :return:
#     """
#
#     if len(np.shape(X)) == 1:
#         return X / np.abs(X)
#     else:
#         a = np.shape(X)[1]
#         b = np.shape(X)[0]
#         return np.dot(np.reshape(np.transpose(
#             np.sqrt(1 / somme_colonnes(np.transpose(X ** 2)))), (b, 1)),
#             np.ones((1, a))) * X


def calcvertex_normals(FV, N):
    """
    CalcVertexNormals calculates the normals and voronoi areas at each vertex
    INPUT:
    FV - triangle mesh in face vertex structure
    N - face normals
    OUTPUT -
    VertexNormals - [Nv X 3] matrix of normals at each vertex
    Avertex - [NvX1] voronoi area at each vertex
    Acorner - [NfX3] slice of the voronoi area at each face corner
    """

    print("Calculating vertex normals .... Please wait")

    "Get all the edge vectors"
    e0 = np.array(FV.vertices[FV.faces[:, 2], :] -
                  FV.vertices[FV.faces[:, 1], :])
    e1 = np.array(FV.vertices[FV.faces[:, 0], :] -
                  FV.vertices[FV.faces[:, 2], :])
    e2 = np.array(FV.vertices[FV.faces[:, 1], :] -
                  FV.vertices[FV.faces[:, 0], :])

    "Normalize edge vectors "
    e0_norm = tut.unitize(e0)
    e1_norm = tut.unitize(e1)
    e2_norm = tut.unitize(e2)

    de0 = np.sqrt((e0[:, 0]) ** 2 + (e0[:, 1]) ** 2 + (e0[:, 2]) ** 2)
    de1 = np.sqrt((e1[:, 0]) ** 2 + (e1[:, 1]) ** 2 + (e1[:, 2]) ** 2)
    de2 = np.sqrt((e2[:, 0]) ** 2 + (e2[:, 1]) ** 2 + (e2[:, 2]) ** 2)
    l2 = np.array([de0 ** 2, de1 ** 2, de2 ** 2])
    l2 = np.transpose(l2)
    """
    using ew to compute the cot of the angles for the voronoi area calculation
    ew is the triangle barycenter.
    We check later if it's inside or outside the triangle
    """
    ew = np.array(
        [
            l2[:, 0] * (l2[:, 1] + l2[:, 2] - l2[:, 0]),
            l2[:, 1] * (l2[:, 2] + l2[:, 0] - l2[:, 1]),
            l2[:, 2] * (l2[:, 0] + l2[:, 1] - l2[:, 2]),
        ]
    )
    s = (de0 + de1 + de2) / 2

    "Af - face area vector"
    Af = np.sqrt(s * (s - de0) * (s - de1) * (s - de2))

    "herons formula for triangle area, "
    "could have also used  0.5 * norm(cross(e0,e1)) "
    "Calc weights"
    Acorner = np.zeros((np.shape(FV.faces)[0], 3))
    Avertex = np.zeros((np.shape(FV.vertices)[0], 1))

    "Compute vertex normals"
    VertexNormals, up, vp = (
        np.zeros((np.shape(FV.vertices)[0], 3)),
        np.zeros((np.shape(FV.vertices)[0], 3)),
        np.zeros((np.shape(FV.vertices)[0], 3)),
    )

    for i in range(np.shape(FV.faces)[0]):
        wfv1 = Af[i] / ((de1[i] ** 2) * (de2[i] ** 2))
        wfv2 = Af[i] / ((de0[i] ** 2) * (de2[i] ** 2))
        wfv3 = Af[i] / ((de1[i] ** 2) * (de0[i] ** 2))

        VertexNormals[FV.faces[i][0], :] += wfv1 * N[i, :]
        VertexNormals[FV.faces[i][1], :] += wfv2 * N[i, :]
        VertexNormals[FV.faces[i][2], :] += wfv3 * N[i, :]

        """
        Calculate areas for weights according to Mayar et al. [2002]
        Check if the triangle is obtuse, right or acute
        """
        "Changed shape for ew"

        if ew[0][i] <= 0:
            Acorner[i][1] = (
                -0.25 * l2[i][2] * Af[i] /
                (np.dot(e0[i, :], np.transpose(e2[i, :])))
            )
            Acorner[i][2] = (
                -0.25 * l2[i][1] * Af[i] /
                (np.dot(e0[i, :], np.transpose(e1[i, :])))
            )
            Acorner[i][0] = Af[i] - Acorner[i][2] - Acorner[i][1]
        elif ew[1][i] <= 0:
            Acorner[i][2] = (
                -0.25 * l2[i][0] * Af[i] /
                (np.dot(e1[i, :], np.transpose(e0[i, :])))
            )
            Acorner[i][0] = (
                -0.25 * l2[i][2] * Af[i] /
                (np.dot(e1[i, :], np.transpose(e2[i, :])))
            )
            Acorner[i][1] = Af[i] - Acorner[i][2] - Acorner[i][0]
        elif ew[2][i] <= 0:
            Acorner[i][0] = (
                -0.25 * l2[i][1] * Af[i] /
                (np.dot(e2[i, :], np.transpose(e1[i, :])))
            )
            Acorner[i][1] = (
                -0.25 * l2[i][0] * Af[i] /
                (np.dot(e2[i, :], np.transpose(e0[i, :])))
            )
            Acorner[i][2] = Af[i] - Acorner[i][1] - Acorner[i][0]
        else:
            ewscale = 0.5 * Af[i] / (ew[0][i] + ew[1][i] + ew[2][i])
            Acorner[i][0] = ewscale * (ew[1][i] + ew[2][i])
            Acorner[i][1] = ewscale * (ew[0][i] + ew[2][i])
            Acorner[i][2] = ewscale * (ew[1][i] + ew[0][i])

        Avertex[FV.faces[i][0]] += Acorner[i][0]
        Avertex[FV.faces[i][1]] += Acorner[i][1]
        Avertex[FV.faces[i][2]] += Acorner[i][2]

        " Calcul initial coordinate system "
        up[FV.faces[i][0], :] = e2_norm[i, :]
        up[FV.faces[i][1], :] = e0_norm[i, :]
        up[FV.faces[i][2], :] = e1_norm[i, :]

    VertexNormals = tut.unitize(VertexNormals)

    " Calcul initial vertex coordinate system"

    for i in range(np.shape(FV.vertices)[0]):
        up[i, :] = np.cross(up[i, :], VertexNormals[i, :])
        up[i, :] = up[i, :] / np.linalg.norm(up[i, :])
        vp[i, :] = np.cross(VertexNormals[i, :], up[i, :])

    print("Finished calculating vertex normals")
    return VertexNormals, Avertex, Acorner, up, vp


def principal_curvatures(FV, VertexSFM, up, vp):
    """
    Calculates the principal curvatures and prncipal directions
    :param FV: triangular mesh
    :param VertexSFM: second fundemental matrix for each vertex
    :param up: vertex local coordinate frame
    :param vp: vertex local coordinate frame
    :return:
    PrincipalCurvature : Matrix containing pricipale curvatures
    ( dim = 2 * Number of vertices )
    PrincipalDi1 , PrincipalDi2 : First and second principal directions
    """

    print("Calculating Principal Components ... Please wait")

    "Calculate principal curvatures"
    PrincipalCurvature = np.zeros((2, np.shape(FV.vertices)[0]))
    PrincipalDi1, PrincipalDi2 = [
        np.zeros((np.shape(FV.vertices)[0], 3)),
        np.zeros((np.shape(FV.vertices)[0], 3)),
    ]
    for i in range(np.shape(FV.vertices)[0]):
        npp = np.cross(up[i, :], vp[i, :])
        r_old_u, r_old_v = rotate_coordinate_system(up[i, :], vp[i, :], npp)
        ku = VertexSFM[i][0][0]
        kuv = VertexSFM[i][0][1]
        kv = VertexSFM[i][1][1]
        c, s, tt = 1, 0, 0
        if kuv != 0:
            "Jacobi rotation to diagonalize"
            h = 0.5 * (kv - ku) / kuv
            if h < 0:
                tt = 1 / (h - np.sqrt(1 + h ** 2))
            else:
                tt = 1 / (h + np.sqrt(1 + h ** 2))
            c = 1 / np.sqrt(1 + tt ** 2)
            s = tt * c
        k1 = ku - tt * kuv
        k2 = kv + tt * kuv
        if abs(k1) >= abs(k2):
            PrincipalDi1[i, :] = c * r_old_u - s * r_old_v
        else:
            [k1, k2] = [k2, k1]
            PrincipalDi1[i, :] = s * r_old_u + c * r_old_v
        PrincipalDi2[i, :] = np.cross(npp, PrincipalDi1[i, :])
        PrincipalCurvature[0][i] = k1
        PrincipalCurvature[1][i] = k2

        if np.isnan(k1) or np.isnan(k2):
            print("Nan")

    print("Finished Calculating principal components")
    return PrincipalCurvature, PrincipalDi1, PrincipalDi2


def rotate_coordinate_system(up, vp, nf):
    """
    RotateCoordinateSystem performs the rotation of the vectors up and vp
    to the plane defined by nf as its normal vector
    :param up: vectors to be rotated (vertex coordinate system)
    :param vp: vectors to be rotated (vertex coordinate system)
    :param nf: face normal
    :return:
    r_new_u,r_new_v : new rotated vectors
    """

    r_new_u = up
    r_new_v = vp
    npp = np.cross(up, vp) / np.linalg.norm(np.cross(up, vp))
    ndot = np.dot(nf, np.transpose(npp))
    if ndot <= -1:
        r_new_u = -r_new_u
        r_new_v = -r_new_v
    perp = nf - ndot * npp
    dperp = (npp + nf) / (1 + ndot)
    r_new_u = r_new_u - dperp * np.dot(perp, np.transpose(r_new_u))
    r_new_v = r_new_v - dperp * np.dot(perp, np.transpose(r_new_v))

    return r_new_u, r_new_v


def decompose_curvature(in_curv):
    """
    Decompose the principal curvatures of a mesh in ShapeIndex and curvedness
    Implementation based on
    ' Surface shape and curvature scales
      Jan JKoenderink & Andrea Jvan Doorn
      Image and Vision Computing
      Volume 10, Issue 8, October 1992, Pages 557-564 '
    """
    curvatures = np.array(
        (np.maximum(in_curv[0], in_curv[1]),
         np.minimum(in_curv[0], in_curv[1]))
    )
    shapeIndex = (2 / np.pi) * np.arctan(
        (curvatures[0, :] + curvatures[1, :]) /
        (curvatures[1, :] - curvatures[0, :])
    )
    curvedness = np.sqrt((curvatures[0, :] ** 2 + curvatures[1, :] ** 2) / 2)
    return shapeIndex, curvedness


def curvedness_shapeIndex(mesh):
    curv = curvatures_and_derivatives(mesh)[0]
    return decompose_curvature(curv)


# def somme_colonnes(X):
#     """
#
#     INPUT :
#     X : A matrix with any dimension
#     OUTPUT:
#     A row that contains the sum of each column
#     """
#     xx = list()
#     for i in range(np.shape(X)[1]):
#         xx.append(sum(X[:, i]))
#     return np.array(xx)
