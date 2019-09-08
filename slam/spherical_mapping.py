import numpy as np
import trimesh


def mobius_transformation(a, b, c, d, plane_mesh):
    """
    see https://en.wikipedia.org/wiki/M%C3%B6bius_transformation
    :param a:Complex
    :param b:Complex
    :param c:Complex
    :param d:Complex
    :param plane_mesh: trimesh mesh
    :return:
    """
    array_complex = plane_mesh.vertices[:, 0] + \
        1.0j * plane_mesh.vertices[:, 1]
    numerator = (a * array_complex) + b
    denominator = (c * array_complex) + d

    transformed_complex_plane = numerator / denominator
    transformed_vertices = np.array([transformed_complex_plane.real,
                                     transformed_complex_plane.imag,
                                     plane_mesh.vertices[:, 2]]).T.copy()
    transformed_plane_mesh = \
        trimesh.Trimesh(vertices=transformed_vertices,
                        faces=plane_mesh.faces.copy(),
                        process=False)
    return transformed_plane_mesh


def stereo_projection(sphere_mesh, h=None, invert=True):
    """
    compute the stereographic projection from the unit sphere (center = 0,
    radius = 1) onto the horizontal plane which 3rd coordinate is h of the
    vertices given
    :param sphere_mesh: trimesh spherical mesh to be projected onto the plane
    :param h: 3rd coordinate of the projection plane
    :param invert: Boolean value to invert output mesh faces orientation
    upwards
    :return: trimesh planar mesh, 3rd coordinate is equal to h
    """
    vertices = sphere_mesh.vertices.copy()
    if h is None:
        h = -1
    for ind, vert in enumerate(vertices):
        vertices[ind, 0] = (-h + 1) * vert[0] / (1 - vert[2])
        vertices[ind, 1] = (-h + 1) * vert[1] / (1 - vert[2])
        vertices[ind, 2] = h

    plane_mesh = trimesh.Trimesh(vertices=vertices,
                                 faces=sphere_mesh.faces.copy(),
                                 process=False)
    if invert:
        plane_mesh.invert()
    return plane_mesh


def inverse_stereo_projection(plane_mesh, h=None, invert=True):
    """
    compute the inverse stereograhic projection from an horizontal plane onto
    the unit sphere (center = 0, radius = 1)
    :param plane_mesh: trimesh planar mesh to be inverse projected onto the
    sphere
    :param h: 3rd coordinate of the projection plane
    :param invert: Boolean value to invert input mesh faces orientation upwards
    to be consistent with stereo_projection
    :return: trimesh unit sphere from inverse projected plane_mesh
    """
    if invert:
        plane_mesh.invert()
    vertices = plane_mesh.vertices.copy()
    if h is None:
        h = vertices[0, 2]
    for ind, vert in enumerate(vertices):
        denom = ((1 - h) ** 2 + vert[0] ** 2 + vert[1] ** 2)
        vertices[ind, 2] = (-(1 - h) ** 2 + vert[0] ** 2 + vert[1] ** 2)\
            / denom
        vertices[ind, 1] = 2 * (1 - h) * vert[1] / denom
        vertices[ind, 0] = 2 * (1 - h) * vert[0] / denom

    sphere_mesh = trimesh.Trimesh(vertices=vertices,
                                  faces=plane_mesh.faces.copy(),
                                  process=False)
    return sphere_mesh
