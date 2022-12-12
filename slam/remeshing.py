"""
Tools for remeshing an input mesh
"""


def spherical_interpolation_nearest_neigbhor(
    source_spherical_mesh, target_spherical_mesh, info_to_interpolate
):
    """
    nearest neighbor interpolation between two spheres
    For each vertex of target_spherical_mesh, find the nearest one
    in source_spherical_mesh and use this vertex-level correspondence
    to pick values in info_to_interpolate
    :param source_spherical_mesh: spherical Trimesh object
    :param target_spherical_mesh: spherical Trimesh object
    :param info_to_interpolate: vector with shape[0] equal to the
    number of vertices in source_spherical_mesh
    :return: interpolated info_to_interpolate
    """
    # This line would interpolate the reverse way (each vertex of
    # source_spherical_mesh, find the nearest one in target_spherical_mesh
    # distance, index =
    # target_spherical_mesh.kdtree.query(source_spherical_mesh.vertices)

    # import time
    # t0 = time.time()

    # the use of kdtree from trimesh is ~100x faster than the loop hereafter
    distance, index = source_spherical_mesh.kdtree.query(
        target_spherical_mesh.vertices)

    # t1 = time.time()
    # source_vertex_number = source_spherical_mesh.vertices.shape[0]
    # nn_corresp = []
    # for v in target_spherical_mesh.vertices:
    #     nn_tmp = np.argmin(np.sum(np.square(
    #     np.tile(v, (source_vertex_number, 1))
    #     - source_spherical_mesh.vertices), 1))
    #     nn_corresp.append(nn_tmp)
    # t2 = time.time()
    # print('with kdtree :'+str(t1-t0))
    # print('with loop :'+str(t2-t1))

    return info_to_interpolate[index]
