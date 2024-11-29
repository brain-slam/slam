import numpy as np
from scipy.spatial.distance import cdist
import trimesh
import trimesh.intersections
import trimesh.triangles
import slam.geodesics
import slam.utils as utils


def cortical_surface_profiling(mesh, rot_angle, r_step, max_samples):
    """
    Surface profiling for a given cortical surface.
    NOTE:
    This function returns 2D profiling coordinates directly instead of 3D.
    These 2D points are used to generate the feature Maps.
    :param mesh: trimesh object
        The cortical surface mesh.
    :param rot_angle: float
        Degree of rotation angle.
    :param r_step: float
        Length of sampling steps.
    :param max_samples:
        Maximum of samples in one profiles.
    :return: (N_vertex, N_p, N_s) float
        Profiles points in their 2D coordinates.
        (x, y) respectively
    """

    vert = mesh.vertices
    poly = mesh.faces.astype(np.int32)
    norm = mesh.vertex_normals

    # compute the geodesic map of cortical surface within the specified radius
    # NOTE: This needs some time
    area_radius = r_step * max_samples * 2
    area_geodist = slam.geodesics.local_gdist_matrix(mesh, area_radius)

    #
    profile_samples_x = []
    profile_samples_y = []
    length = len(vert)

    for i in range(length):
        # for every vertex, do the surface profiling.
        vert_i = vert[i]
        vert_norm_i = norm[i]

        # limit the intersection area into the area_radius (on Distmap) from
        # center
        vert_distmap = area_geodist[i].toarray()[0]
        area_geodist_v = np.where(vert_distmap > 0)[0]
        area_geodist_faces = vert2poly_indices(area_geodist_v, poly)
        intersect_mesh = mesh.submesh(np.array([area_geodist_faces]))[0]

        # randomly select initial direction of rotation R0
        init_rot_dir = np.array([1, 1, 1]) - vert_i

        # get the profile samplings on surface
        sam_prof = surface_profiling_vert(
            vert_i,
            vert_norm_i,
            init_rot_dir,
            rot_angle,
            r_step,
            max_samples,
            intersect_mesh,
        )

        # compute the 2D coordinates (x, y) for all profile points
        sample_x, sample_y = compute_profile_coord_x_y(
            sam_prof, vert[i], norm[i])

        profile_samples_x.append(sample_x)
        profile_samples_y.append(sample_y)

    return np.array(profile_samples_x), np.array(profile_samples_y)


def surface_profiling_vert(
    vertex, vert_norm, init_rot_dir, rot_angle, r_step, max_samples, mesh
):
    """
    Implement the profile sampling process for a given vertex.
    NOTE:
    For a given vertex,
        Number of profiles N_p = 360/theta
    For each profiles,
        Number of Sampling points N_s = max_samples
    :param vertex: (3,) float
        Target vertex (center vertex)
    :param vert_norm: (3,) float
        Vertex normal
    :param init_rot_dir: (3, )
        Initial direction of rotation R0
    :param rot_angle: (3,) float
        Degree of rotation angle
    :param r_step: float
        Length of sampling steps
    :param max_samples: int
        Maximum of samples in one profiles
    :param mesh: trimesh object
        Intersecting mesh
    :return: (N_p, N_s, 3) float
        Profiles points in 3D coordinates.
    """

    profile_list = []  # record all the samples x and y
    vertex = np.array(vertex)

    # project the dir_R0 onto the tangent plane
    rot_vec0 = utils.project_vector2tangent_plane(vert_norm, init_rot_dir)[0]
    round_angle = 360

    for i in range(int(round_angle / rot_angle)):
        # set the rotation directions
        rot_angle_alpha = (i * rot_angle) * 1.0 / 360 * 2 * np.pi
        rot_mat_alpha = utils.get_rotate_matrix(vert_norm, rot_angle_alpha)
        rot_vec_alpha = np.dot(rot_vec0, rot_mat_alpha)
        p_norm = np.cross(vert_norm, rot_vec_alpha)

        # Get the intersection lines
        # the lines contains the rotation direction and the reverse one.
        intersect_lines = trimesh.intersections.mesh_plane(
            mesh, p_norm, vertex)

        # Select the points in the direction of rotation vector
        points_i, _, _ = select_points_orientation(
            intersect_lines, rot_vec_alpha, vertex, vert_norm
        )

        # Calculate the samples of profiles
        points_profile = compute_profiles_sampling_points(
            points_i, vertex, max_samples, r_step
        )

        profile_list.append(points_profile)

    return np.array(profile_list)


def second_round_profiling_vert(
    vertex,
    vert_norm,
    init_rot_dir,
    rot_angle,
    r_step,
    max_samples,
    mesh,
    mesh_face_index,
):
    """
    Implement the profile sampling process to get the feature values of each
    profiling points.
    The function name comes from the description of the method in the article。
    Different from the surface_profiling_vert, the mesh_face_index is
    obligatory.
    :param vertex: (3,) float
        Target vertex (center vertex)
    :param vert_norm: (3,) float
        Vertex normal
    :param init_rot_dir: (3, )
        Initial direction of rotation R0
    :param rot_angle: (3,) float
        Degree of rotation angle
    :param r_step: float
        Length of sampling steps
    :param max_samples: int
        Maximum of samples in one profiles
    :param mesh: trimesh object
        Intersecting mesh
    :param mesh_face_index:
        Indices of polygons of mesh.
        Use to record which polygon the sampling points belongs to.
    :return:
        profile_points: (N_p, N_s, 3, 3) float
            For each profile points contain [p1, p2, sample_points],
            where p1, p2 are the points used to calculate the sampling points.
        profile_intersect_faces: ((N_p, N_s,) int
    """

    profile_points = []  # record all the profiling points and interpolation
    # points
    profile_intersect_faces = []  # record all the faces id that contain the
    # sample points
    vertex = np.array(vertex)

    # project the dir_R0 onto the tangent plane
    rot_vec0 = utils.project_vector2tangent_plane(vert_norm, init_rot_dir)[0]
    round_angle = 360

    for i in range(int(round_angle / rot_angle)):
        # set the rotation directions
        rot_angle_alpha = (i * rot_angle) * 1.0 / 360 * 2 * np.pi
        rot_mat_alpha = slam.utils.get_rotate_matrix(
            vert_norm, rot_angle_alpha)
        rot_vec_alpha = np.dot(rot_vec0, rot_mat_alpha)
        p_norm = np.cross(vert_norm, rot_vec_alpha)

        # Get the intersection lines
        # the lines contains the rotation direction and the reverse one.
        intersect_lines, faces = trimesh.intersections.mesh_plane(
            mesh, p_norm, vertex, return_faces=True
        )

        # get the global index of faces
        intersect_fm_index = mesh_face_index[faces]

        # Select the points in the direction of rotation vector
        orient_points_i, orient_p_id, ori_lines_id = select_points_orientation(
            intersect_lines, rot_vec_alpha, vertex, vert_norm
        )

        orient_face_id = intersect_fm_index[ori_lines_id]

        # Calculate the samples of profiles
        points_interp_profile, cor_faces_index = compute_profiles_sampling_points(
            orient_points_i, vertex, max_samples, r_step, orient_face_id
        )

        profile_points.append(points_interp_profile)
        profile_intersect_faces.append(cor_faces_index)

    return np.array(profile_points), np.array(profile_intersect_faces)


def compute_profiles_sampling_points(
    points_intersect, origin, max_samples, r_step, face_id=None
):
    """
    Calculate the sampling points on each profiles.
    :param points_intersect: (n, 3) float
    :param origin: (3,) float
        origin vertex
    :param max_samples: int
        Maximum of samples in one profiles
    :param r_step: float
        Length of sampling steps
    :param face_id: (n,) int
        Indices of polygons which intersecting points belong to.
        Default is None, it is only used in the second round profiling.
    :return:
        When face_id is None, return
            sampling points on profiles: (n, 3) float
        Otherwise,
            points_interpolate_profile: (n, 3, 3) float
                contains [p1, p2, sample_points]
                where p1, p2 are the points used to calculate the sampling
                points.
            cor_faces_index: (n,)
                the corresponding faces of profile points
    """

    if len(points_intersect) == 0:
        # the origin point is out of intersection points
        profile_points = list(np.zeros([max_samples, 3]))
        return profile_points

    # record the length of segment
    length_sum = np.linalg.norm(points_intersect[0] - origin)
    minued_lenth = 0
    count_i = 0
    # record the i when the sample distance firstly exceed the maximum
    # of intersection
    exceed_index = 0
    exceed_bool = True
    count_max = len(points_intersect)

    profile_points = []
    # Record the two end-points and sampling points
    points_interpolate_profile = []
    # the corresponding faces
    cor_faces_index = []

    for j in range(max_samples):
        sample_dist = (j + 1) * r_step
        if sample_dist <= length_sum and count_i == 0:
            # the point is between center point and the closest one
            point0 = origin
            point1 = points_intersect[0]

        elif sample_dist <= length_sum and count_i != 0:
            point0 = points_intersect[count_i - 1]
            point1 = points_intersect[count_i]

        else:
            minued_lenth = length_sum
            count_i += 1

            # the distance of sample exceed the local maximum
            if count_i == count_max:
                # first time arrive at boundary
                if exceed_bool:
                    exceed_index = j
                    exceed_bool = False
                count_i -= 1
                sample_dist = (exceed_index + 1) * r_step
                point0 = points_intersect[count_i - 1]
                point1 = points_intersect[count_i]
            else:
                point0 = points_intersect[count_i - 1]
                point1 = points_intersect[count_i]
                length_sum += np.linalg.norm(point1 - point0)

        if np.linalg.norm(point1 - point0) == 0:
            alpha = 0
        else:
            alpha = (sample_dist - minued_lenth) / \
                np.linalg.norm(point1 - point0)

        sample_point = (1 - alpha) * point0 + alpha * point1

        profile_points.append(sample_point)
        points_interpolate_profile.append([point0, point1, sample_point])
        # save the related intersect mesh faces
        if face_id is not None:
            cor_faces_index.append(face_id[count_i])

    if face_id is None:
        return profile_points

    return points_interpolate_profile, cor_faces_index


def vert2poly_indices(vertex_array, poly_array):
    """
    Find vertex-polygon indices from the polygons array of vertices
    TODO There is a func in the lastest trimesh:
        trimesh.geometry.vertex_face_indices()
    :param vertex_array:
    :param poly_array:
    :return:
    """

    vert_poly_arr = np.array([], dtype=int)
    for i in range(len(vertex_array)):
        poly_i = np.where(poly_array == vertex_array[i])[0]
        vert_poly_arr = np.hstack((vert_poly_arr, poly_i))

    return np.unique(vert_poly_arr)


def select_points_orientation(intersect_points, r_alpha, origin, norm):
    """
    Select points in a specified orientation,
    and ordered them by distance from the center.
    :param intersect_points: (n, 2, 3) float
        Points of intersecting lines.
    :param r_alpha: (3,) float
        Orientation vector
    :param origin: (3,) float
        Origin point
    :param norm: (3,) float
        Normal of origin point
    :return: orient_points,    (n, 3) float
             orient_p_indices, (n,) int
             lines_indices,    (n,) int
        Ordered points in the orientation.
    """

    points_i = intersect_points.reshape(intersect_points.size // 3, 3)

    # find the center points
    p_idx, count_coord = np.unique(
        np.where(points_i == origin)[0], return_counts=True)
    origin_index = p_idx[np.where(count_coord == 3)[0]]

    if len(origin_index) == 0:
        # the intersection result exclude origin point
        orient_points = []
        orient_p_indices = []
        lines_indices = []
        return orient_points, orient_p_indices, lines_indices

    ordered_points, ordered_p_indices = radial_sort(points_i, origin, norm)

    orientation_vec = np.dot(ordered_points - origin, r_alpha)
    orient_point_idx = np.where(orientation_vec > 0)[0][::2]
    orient_points = ordered_points[orient_point_idx]
    orient_p_indices = ordered_p_indices[orient_point_idx]

    # find the closest point
    p2o_len = cdist(orient_points, np.array([origin]), metric="euclidean")
    p2o_len = p2o_len.reshape(p2o_len.shape[0])
    ordered_p2o = np.argsort(p2o_len)
    orient_points = orient_points[ordered_p2o]
    orient_p_indices = orient_p_indices[ordered_p2o]

    # get the ordered intersection lines
    ori_lines_indices = orient_p_indices // 2

    return orient_points, orient_p_indices, ori_lines_indices


def radial_sort(points, origin, normal):
    """
    NOTE:
    This function is derived from the
        trimesh.points.radial_sort(points_i, origin, norm)
    I overwrite this function to return both the coordinates and indices of
    the points.
    Sorts a set of points radially (by angle) around an
    an axis specified by origin and normal vector.
    Parameters
    --------------
    points : (n, 3) float
      Points in space
    origin : (3,)  float
      Origin to sort around
    normal : (3,)  float
      Vector to sort around
    Returns
    --------------
    ordered : (n, 3) float
      Same as input points but reordered
    """

    # create two axis perpendicular to each other and the normal,
    # and project the points onto them
    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    pt_vec = points - origin
    pr0 = np.dot(pt_vec, axis0)
    pr1 = np.dot(pt_vec, axis1)

    # calculate the angles of the points on the axis
    angles = np.arctan2(pr0, pr1)

    # return the points and their indices sorted by angle
    return points[(np.argsort(angles))], np.argsort(angles)


def compute_profile_coord_x_y(profile, origin, normal):
    """
    Calculate the 2D coordinates of the profiling points in their rotation
    planes
    These points are used to generate the feature maps mentioned in the
    articles.
    :param profile: (N_p, N_s, 3) float
        Sampling points of profiles in 3D
    :param origin: (3,) float
        Center vertex of profiles
    :param normal: (3,) float
        Normal of origin
    :return: (N_p, N_s) float
        The coordinate x, y
    """
    num_prof = len(profile)
    num_sample = len(profile[0])

    # flat the samples
    profile_samples = profile.reshape([num_prof * num_sample, 3])

    # get the vector
    pro_sam_vec = profile_samples - origin
    vec_x = utils.project_vector2tangent_plane(normal, pro_sam_vec)
    vec_y = utils.project_vector2vector(normal, pro_sam_vec)

    # the length of the vector that projected onto the normal vector
    # x
    length_x = np.linalg.norm(vec_x, axis=1)
    x = length_x.reshape([num_prof, num_sample])

    # y
    sign_y = np.sign((profile_samples - origin).dot(normal))
    length_y = np.linalg.norm(vec_y, axis=1) * sign_y
    y = length_y.reshape([num_prof, num_sample])

    return x, y


def get_texture_value_on_profile(
    texture, mesh, profiling_samples, profiling_samples_fid
):
    """
    Calculate the texture values of each points on profiles by barycentric
    interpolation
    :param texture: slam texture
    :param mesh: trimesh object
    :param profiling_samples: (N, N_p, N_s, 3, 3) float
        N = Number of center vertices for surface profiling.
        N_p = Number of profiles for each center.
        N_s = Number of sampling points on each profiles.
        3 = [p1, p2, sampling points]
        3 = (3,) float
    :param profiling_samples_fid: (N, N_p, N_s) int
        Faces id corresponding to the profile sampling points.
    :return:
        texture_profile: (N, N_p, N_s) float
    """

    # compute the barycentric parameters of each profile point to its
    # co-faces of mesh
    barycentric_coord = compute_profile_barycentric_para(
        profiling_samples, mesh, profiling_samples_fid
    )

    # compute the features of each profile
    tex_arr = texture.darray[0]
    texture_profile = compute_profile_texture_barycentric(
        tex_arr, mesh, profiling_samples_fid, barycentric_coord
    )

    return texture_profile


def compute_profile_barycentric_para(profile_sample_points, mesh, triangle_id):
    """
    Compute the barycentric parameters of each points on profiles
    :param profile_sample_points: (N, N_p, N_s, 3, 3) float
        N = Number of center vertices for surface profiling.
        N_p = Number of profiles for each center.
        N_s = Number of sampling points on each profiles.
        3 = [p1, p2, sampling points]
        3 = (3,) float
    :param mesh: trimesh object
    :param triangle_id: (N, N_p, N_s) int
        Faces id corresponding to the profile sampling points.
    :return:
        barycentric: (N, N_p, N_s, 3) float
        Barycentric coordinates for all profiles points
    """

    if len(profile_sample_points.shape) != 5:
        raise Exception(
            "Wrong type of profile_sample_points, " "it must be (N, N_p, N_s, 3, 3)."
        )

    vert = mesh.vertices
    poly = mesh.faces

    # get the sample points on profile
    sample_points_profile = profile_sample_points[:, :, :, 2]
    sample_points = sample_points_profile.reshape(
        sample_points_profile.size // 3, 3)

    # get the faces
    triangle_id = triangle_id.reshape(triangle_id.size)
    triangles_v = vert[poly[triangle_id]]

    barycentric = trimesh.triangles.points_to_barycentric(
        triangles_v, sample_points)
    barycentric = barycentric.reshape(
        len(sample_points_profile),
        len(sample_points_profile[0]),
        len(sample_points_profile[0][0]),
        3,
    )

    return barycentric


def compute_profile_texture_barycentric(
        texture, mesh, triangle_id, barycentric_coord):
    """
    Compute the texture values of each points on profiles
    :param texture: darray of slam texture
    :param mesh: trimesh object
    :param triangle_id: (N, N_p, N_s) int
        Faces id corresponding to the profile sampling points.
    :param barycentric_coord: (N, N_p, N_s, 3) float
         Barycentric coordinates for all profiles points
    :return:
    """

    num_profiles = len(barycentric_coord)
    num_areas = len(barycentric_coord[0])
    num_sides = len(barycentric_coord[0][0])
    poly = mesh.faces

    triangle_id = triangle_id.reshape(triangle_id.size)
    barycentric = barycentric_coord.reshape(barycentric_coord.size // 3, 3)

    feature_tri_points = texture[poly[triangle_id]]
    texture_profile = np.dot(feature_tri_points * barycentric, [1, 1, 1])

    return texture_profile.reshape(num_profiles, num_areas, num_sides)
