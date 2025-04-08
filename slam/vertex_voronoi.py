import numpy as np
import itertools


def vertex_voronoi(mesh):
    """
    Adapted from method of Maxime Dieudonné maxime.dieudonne@univ-amu.fr which used pandas package
    compute vertex voronoi of a mesh as described in
    Meyer, M., Desbrun, M., Schroder, P., Barr, A. (2002).
    Discrete differential geometry operators for triangulated 2manifolds.
    Visualization and Mathematics, 1..26.
    :param mesh: trimesh object
    :return: numpy array of shape (mesh.vertices.shape[0],)
    """
    Nbp = mesh.faces.shape[0]  # Number of polygon faces
    obt_angs = mesh.face_angles > np.pi / 2
    obt_poly = obt_angs[:, 0] | obt_angs[:, 1] | obt_angs[:, 2]
    print('    -percent polygon with obtuse angle ', 100.0 * len(np.where(obt_poly)[0]) / Nbp)
    cot = 1 / np.tan(mesh.face_angles)
    #
    # Extract the segments of the faces (pairs of vertices)
    # Comb contains the vertex indices of the three segments of each face (Nbp*3,2)
    comb = np.array([list(itertools.combinations(mesh.faces[i], 2)) for i in range(Nbp)])
    comb = comb.reshape([comb.size // 2, 2])
    #
    # Extract the coordinates of the vertices of the segments: V1 for first column, V2 for the second
    V1 = np.array([mesh.vertices[i] for i in comb[:, 0]])
    V2 = np.array([mesh.vertices[i] for i in comb[:, 1]])
    #
    # Dist array : compute the square of the norm of each segments. We need to switch the first and the last colums to gets
    # the right order : D0, D1, D2
    # faces | segments of the faces
    # F0    | D0 D1 D2
    # F1    | D0 D1 D2
    # FN    | D0 D1 D2
    Dist = np.sum(np.power(V1 - V2, 2), axis=1)
    Dist = Dist.reshape([Dist.size // 3, 3])
    Dist[:, [2, 0]] = Dist[:, [0, 2]]
    #
    # VorA0 : compute the voronoi area (hypothesis : the face triangle is Aigu) for all the V0 of the faces
    # VorA1 : compute the voronoi area (hypothesis : the face triangle is Aigu) for all the V1 of the faces
    # VorA2 : compute the voronoi area (hypothesis : the face triangle is Aigu) for all the V2 of the faces
    # VorA : concatenation of VorA0, VorA1, VorA2.
    # faces | segments of the faces
    # F0    | VorA0 VorA1 VorA2
    # F1    | VorA0 VorA1 VorA2
    # FN    | VorA0 VorA1 VorA2
    VorA0 = Dist[:, 1] * cot[:, 1] + Dist[:, 2] * cot[:, 2]
    VorA1 = Dist[:, 0] * cot[:, 0] + Dist[:, 2] * cot[:, 2]
    VorA2 = Dist[:, 0] * cot[:, 0] + Dist[:, 1] * cot[:, 1]
    VorA = np.array([VorA0, VorA1, VorA2]).transpose().flatten()

    face_flat = mesh.faces.flatten()
    area_face = mesh.area_faces
    area_face_df = np.repeat(area_face, 3)
    obt_poly_df = np.repeat(obt_poly, 3)
    obt_angs_flat = obt_angs.flatten()
    #
    # we create 3 arrays according the condition on Fobt and Aobt. The 3 possibilities are :
    #  (Fobt, Aobt) = (False, False) -> the voronoi area of a vertex in 1 triangle face is given in VorA
    #  (Fobt, Aobt) = (True, False)  -> the voronoi area of a vertex in 1 triangle face is given by area_face/4
    #  (Fobt, Aobt) = (True, True)  -> the voronoi area of a vertex in 1 triangle face is given by area_face/2
    # area_VorA : sum of the vornoi area only for vertex with angle in  aigue faces
    # area_VorOA : sum of the vornoi area only for vertex with Aigue angle in  Obtue faces
    # area_VorOO : sum of the vornoi area only for vertex with Obtue angle in  Obtue faces
    # area_VorA
    mask_AAAF = ~obt_poly_df
    vertices = face_flat[mask_AAAF]
    VorA_values = VorA[mask_AAAF]
    # Replace "groupby" vertex and "sum" from pandas
    unique_vertices, inverse_indices = np.unique(vertices, return_inverse=True)
    area_VorA = np.zeros_like(unique_vertices, dtype=np.float64)
    np.add.at(area_VorA, inverse_indices, VorA_values)
    area_VorA /= 8
    area_VorA = np.column_stack((unique_vertices, area_VorA))
    # area_VorOA
    mask_AAOF = obt_poly_df & ~obt_angs_flat
    vertices = face_flat[mask_AAOF]
    area_values = area_face_df[mask_AAOF]
    unique_vertices, inverse_indices = np.unique(vertices, return_inverse=True)
    area_VorOA = np.zeros_like(unique_vertices, dtype=np.float64)
    np.add.at(area_VorOA, inverse_indices, area_values)
    area_VorOA /= 4
    area_VorOA = np.column_stack((unique_vertices, area_VorOA))
    # area_VorOO
    mask_OAOF = obt_poly_df & ~obt_angs_flat
    vertices = face_flat[mask_OAOF]
    area_values = area_face_df[mask_OAOF]
    unique_vertices, inverse_indices = np.unique(vertices, return_inverse=True)
    area_VorOO = np.zeros_like(unique_vertices, dtype=np.float64)
    np.add.at(area_VorOO, inverse_indices, area_values)
    area_VorOO /= 2
    area_VorOO = np.column_stack((unique_vertices, area_VorOO))
    #
    # Then the voronoi area of one vertex in the mesh is given by the sum of all the vornoi area in its neighboored triangle faces
    # so we sum for each vertex the area given in area_VorA,area_VorOA and area_VorOO if the value exist
    # area :
    # | idx Vertex      | VorA  | VorOA | VorOO |
    # | 1               |   0.2 |  1.2  | Nan |  -> sum = area_vor
    # | 2               |   0.7 |  Nan  | Nan |  -> sum
    # | ...             |
    # | 480 562         |                        -> sum
    areas = np.vstack((area_VorA, area_VorOA, area_VorOO))
    unique_vertices, inverse_indices = np.unique(areas[:, 0], return_inverse=True)
    area_Vor = np.zeros_like(unique_vertices, dtype=np.float64)
    np.add.at(area_Vor, inverse_indices, areas[:, 1])

    return area_Vor
