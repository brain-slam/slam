import numpy as np


def vertex_voronoi(mesh):
    """
    compute vertex voronoi of a mesh as described in
    Meyer, M., Desbrun, M., Schroder, P., Barr, A. (2002).
    Discrete differential geometry operators for triangulated 2manifolds.
    Visualization and Mathematics, 1..26.
    :param mesh: trimesh object
    :return: numpy array of shape (mesh.vertices.shape[0],)
    """
    Nbv = mesh.vertices.shape[0]
    Nbp = mesh.faces.shape[0]
    obt_angs = mesh.face_angles > np.pi / 2
    obt_poly = obt_angs[:, 0] | obt_angs[:, 1] | obt_angs[:, 2]
    print(
        "    -percent polygon with obtuse angle ",
        100.0 * len(np.where(obt_poly)[0]) / Nbp,
    )
    cot = 1 / np.tan(mesh.face_angles)
    vert_voronoi = np.zeros(Nbv)
    for ind_p, p in enumerate(mesh.faces):
        if obt_poly[ind_p]:
            obt_verts = p[obt_angs[ind_p, :]]
            vert_voronoi[obt_verts] = (
                vert_voronoi[obt_verts] + mesh.area_faces[ind_p] / 2.0
            )
            non_obt_verts = p[[not x for x in obt_angs[ind_p, :]]]
            vert_voronoi[non_obt_verts] = (
                vert_voronoi[non_obt_verts] + mesh.area_faces[ind_p] / 4.0
            )
        else:
            d0 = np.sum(np.power(mesh.vertices[p[1], :] - mesh.vertices[p[2], :], 2))
            d1 = np.sum(np.power(mesh.vertices[p[2], :] - mesh.vertices[p[0], :], 2))
            d2 = np.sum(np.power(mesh.vertices[p[0], :] - mesh.vertices[p[1], :], 2))
            vert_voronoi[p[0]] = (
                vert_voronoi[p[0]] + (d1 * cot[ind_p, 1] + d2 * cot[ind_p, 2]) / 8.0
            )
            vert_voronoi[p[1]] = (
                vert_voronoi[p[1]] + (d2 * cot[ind_p, 2] + d0 * cot[ind_p, 0]) / 8.0
            )
            vert_voronoi[p[2]] = (
                vert_voronoi[p[2]] + (d0 * cot[ind_p, 0] + d1 * cot[ind_p, 1]) / 8.0
            )

    return vert_voronoi
