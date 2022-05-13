def angle_difference(mesh1, mesh2):
    """
    compute the difference of angle between mesh1 and mesh2
    for each of the three angles of each face
    :param mesh1:
    :param mesh2:
    :return:
    """

    return mesh1.face_angles - mesh2.face_angles


def area_difference(mesh1, mesh2):
    """
        Compare_mesh_angle

        Difference between the faces.

        :param mesh1
        :type trimesh

        :param mesh2
        :type trimesh

        :return
    """

    return mesh1.area_faces - mesh2.area_faces


def edge_length_difference(mesh1, mesh2):
    """
    compute the difference of the length of edges between mesh1 and mesh2
    :param mesh1:
    :param mesh2:
    :return:
    """

    return mesh1.edges_unique_length - mesh2.edges_unique_length
