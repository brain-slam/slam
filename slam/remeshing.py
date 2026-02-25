import numpy as np
sphere_check_tol = 0.001


def normalize_sphere_coords(mesh):
    """
    center and normalize the supposed spherical coordinates in mesh.vertices
    Also check that they are spherical by assessing the min and max of the
    norm of the coordinates of the vertices.
    :param mesh:
    :return: mesh with normalized coordinates
    """
    mesh.vertices -= mesh.center_mass
    vects_norm = np.linalg.norm(mesh.vertices, axis=1)
    sphere_radius = np.max(vects_norm)
    # debug
    # print(sphere_radius)
    # sphere_radius_m = np.min(vects_norm)
    # diff = np.max(vects_norm) - np.min(vects_norm)
    # check the mesh is spherical
    assert np.max(vects_norm) - np.min(vects_norm) < sphere_check_tol
    if sphere_radius > 1 + sphere_check_tol:
        mesh.vertices[:, 0] /= vects_norm
        mesh.vertices[:, 1] /= vects_norm
        mesh.vertices[:, 2] /= vects_norm
        # debug
        # sphere_radius = np.max(np.linalg.norm(mesh.vertices, axis=1))
        # print(sphere_radius)
    return mesh


def spherical_interpolation_nearest_neighbor(
    source_spherical_mesh, target_spherical_mesh, normalize_spheres=False
):
    """
    Nearest neighbor interpolation between two spheres
    For each vertex of target_spherical_mesh, find the nearest one
    in source_spherical_mesh
    :param source_spherical_mesh: spherical Trimesh object
    :param target_spherical_mesh: spherical Trimesh object
    :param normalize_spheres: check that the two spheres are
    centered at (0,0,0) with radius=1
    :return: index: index of corresponding vertex in the target mesh
    """
    # check and normalize the two spheres so that the
    # center is (0, 0, 0) and radius=1
    if normalize_spheres:
        target_spherical_mesh = normalize_sphere_coords(target_spherical_mesh)
        source_spherical_mesh = normalize_sphere_coords(source_spherical_mesh)

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

    return index


def texture_spherical_interpolation_nearest_neighbor(
        source_spherical_mesh,
        target_spherical_mesh,
        texture_to_interpolate,
        normalize_spheres=False):
    """
    Nearest neighbor interpolation between two spheres
    Use the vertex-level correspondence obtained with
    spherical_interpolation_nearest_neighbor to select
    corresponding values in the texture_to_interpolate.
    :param source_spherical_mesh: spherical Trimesh object
    :param target_spherical_mesh: spherical Trimesh object
    :param texture_to_interpolate: vector with shape[0] equal to
    the number of vertices in source_spherical_mesh
    :param normalize_spheres: check that the two spheres are
    centered at (0,0,0) with radius=1
    :return: interpolated texture, with shape[0] equal to
    the number of vertices in target_spherical_mesh
    """
    interp_index = spherical_interpolation_nearest_neighbor(
        source_spherical_mesh, target_spherical_mesh, normalize_spheres)

    return texture_to_interpolate[np.array(interp_index)]
