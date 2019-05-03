

def angle_difference(mesh1, mesh2):

    return mesh1.face_angles - mesh2.face_angles


def area_diffference(mesh1, mesh2):

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
