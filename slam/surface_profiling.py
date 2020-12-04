
import numpy as np
from scipy.spatial.distance import cdist
import slam.geodesics
import slam.utils
import trimesh
import trimesh.intersections


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

        # get the profile samplings on surface
        sam_prof = surface_profiling_vert(
            vert_i,
            vert_norm_i,
            rot_angle,
            r_step,
            max_samples,
            intersect_mesh)

        # compute the 2D coordinates (x, y) for all profile points
        sample_x, sample_y = compute_profile_coord_x_y(
            sam_prof, vert[i], norm[i])

        profile_samples_x.append(sample_x)
        profile_samples_y.append(sample_y)

    return np.array(profile_samples_x), np.array(profile_samples_y)


def surface_profiling_vert(
        vertex,
        vert_norm,
        rot_angle,
        r_step,
        max_samples,
        mesh):
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
    :param rot_angle: (3,) float
        Degree of rotation angle
    :param r_step: float
        Length of sampling steps
    :param max_samples:
        Maximum of samples in one profiles
    :param mesh: trimesh object
        Intersecting mesh
    :return: (N_p, N_s, 3) float
        Profiles points in 3D coordinates.
    """

    profile_list = []  # record all the samples x and y

    vertex = np.array(vertex)

    # set initial rotation direction
    # randomly select R0
    dir_r0 = np.array([1, 1, 1]) - vertex

    # project the dir_R0 onto the tangent plane
    rot_vec0 = project_vector2tangent_plane(vert_norm, dir_r0)[0]
    round_angle = 360

    for i in range(int(round_angle / rot_angle)):

        # set the rotation directions
        rot_angle_alpha = (i * rot_angle) * 1.0 / 360 * 2 * np.pi
        rot_mat_alpha = slam.utils.get_rotate_matrix(
            vert_norm, rot_angle_alpha)
        rot_vec_alpha = np.dot(rot_vec0, rot_mat_alpha)
        p_norm = np.cross(vert_norm, rot_vec_alpha)

        # Get the intersection lines
        # the lines contains two directions, the rotation direction and
        # reverse.
        intersect_lines = trimesh.intersections.mesh_plane(
            mesh, p_norm, vertex)

        # select the points in the direction of rotation vector
        points_i = select_points_orientation(
            intersect_lines, rot_vec_alpha, vertex, vert_norm)

        # Calculate the samples of profiles
        if len(points_i) != 0:
            points_sign = []
            # record the length of segment
            length_sum = np.linalg.norm(points_i[0] - vertex)
            minued_lenth = 0
            count_i = 0
            # record the i when the sample distance firstly exceed the maximum
            # of intersection
            exceed_index = 0
            exceed_bool = True
            count_max = len(points_i)

            for j in range(max_samples):
                sample_dist = (j + 1) * r_step
                if sample_dist <= length_sum and count_i == 0:
                    # the point is between center point and the closest one
                    point0 = vertex
                    points1 = points_i[0]

                elif sample_dist <= length_sum and count_i != 0:
                    point0 = points_i[count_i - 1]
                    points1 = points_i[count_i]

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
                        point0 = points_i[count_i - 1]
                        points1 = points_i[count_i]
                    else:
                        point0 = points_i[count_i - 1]
                        points1 = points_i[count_i]
                        length_sum += np.linalg.norm(points1 - point0)

                if np.linalg.norm(points1 - point0) == 0:
                    alpha = 0
                else:
                    alpha = (sample_dist - minued_lenth) / \
                        np.linalg.norm(points1 - point0)

                sample_point = (1 - alpha) * point0 + alpha * points1
                # use to calculate the sign of sample y
                points_sign.append(sample_point)

        else:
            # the origin point is out of intersection points
            points_sign = list(np.zeros([max_samples, 3]))

        profile_list.append(points_sign)

    return np.array(profile_list)


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


def project_vector2tangent_plane(v_n, v_p):
    """
    calculate the projection vector of v_p onto tangent plane of v_n

    :param v_n: array of (3,)  float
        normal vector of v

    :param v_p: array of (n, 3) float
        vector projected

    :return: v_t (n, 3) float
        projection result
    """

    unitev_n = v_n / np.linalg.norm(v_n)

    coeff_v_pn = np.dot(v_p, unitev_n)

    coeff = coeff_v_pn.reshape([coeff_v_pn.size, 1])

    v_pn = coeff * unitev_n

    v_t = v_p - np.array(v_pn)

    return v_t


def project_vector2vector(v_n, v_p):
    """
    calculate the projection vector of v_p onto v_n,
    v_pn = (v_p dot v_n) / |v_n| * unite vector of (v_n)

    :param v_n: array of (3,)  float
        direction vector

    :param v_p: array of (n, 3) float
        vectors projected

    :return: (n, 3) float
        projection result

    """

    unitev_n = v_n / np.linalg.norm(v_n)

    coeff_v_pn = np.dot(v_p, unitev_n)

    coeff = coeff_v_pn.reshape([coeff_v_pn.size, 1])

    v_pn = coeff * unitev_n

    return v_pn


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

    :return: r_points, (2n, 3) float
        Ordered points in the orientation.

    """

    points_i = intersect_points.reshape(intersect_points.size // 3, 3)
    ordered_points = trimesh.points.radial_sort(points_i, origin, norm)

    orientation_vec = np.dot(ordered_points - origin, r_alpha)
    orient_point_idx = np.where(orientation_vec > 0)[0][::2]
    orient_points = ordered_points[orient_point_idx]

    # find the closest point
    p2o_len = cdist(orient_points, np.array([origin]), metric='euclidean')
    p2o_len = p2o_len.reshape(p2o_len.shape[0])
    ordered_p2o = np.argsort(p2o_len)
    orient_points = orient_points[ordered_p2o]

    return orient_points


def compute_profile_coord_x_y(profile, origin, normal):
    """
    Calculate the 2D coordinates of the profiling points in their rotation planes

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
    vec_x = project_vector2tangent_plane(normal, pro_sam_vec)
    vec_y = project_vector2vector(normal, pro_sam_vec)

    # the length of the vector that projected onto the normal vector
    # x
    length_x = np.linalg.norm(vec_x, axis=1)
    x = length_x.reshape([num_prof, num_sample])

    # y
    sign_y = np.sign((profile_samples - origin).dot(normal))
    length_y = np.linalg.norm(vec_y, axis=1) * sign_y
    y = length_y.reshape([num_prof, num_sample])

    return x, y
