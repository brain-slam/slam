
import numpy as np
import slam.geodesics
import trimesh


def cortical_surface_profiling(mesh, theta, r_step, m, save_path):
    """

    :param mesh:
    :param theta:
    :param r_step:
    :param m:
    :param save_path:
    :return:
    """

    vert = mesh.vertices
    poly = mesh.faces.astype(np.int32)
    norm = mesh.vertex_normals

    # compute the geodesic map of cortical surface within the specified radius
    # NOTE: This needs some time
    area_radius = r_step * m * 2
    area_geodist = geodesics.local_gdist_matrix(mesh, area_radius)

    #
    profile_samples_x = []
    profile_samples_y = []
    length = len(vert)

    for i in range(length):
        # for every vertex, do the surface profiling.
        vert_i = vert[i]
        vert_norm_i = norm[i]

        # limit the intersection area into the area_radius (on Distmap) from center
        # TODO There is a func in the lastest trimesh: trimesh.geometry.vertex_face_indices()
        vert_distmap = area_geodist[i].toarray()[0]
        area_geodist_v = np.where(vert_distmap > 0)[0]
        area_geodist_faces = vert2poly_indices(area_geodist_v, poly)
        intersect_mesh = mesh.submesh(np.array([area_geodist_faces]))[0]

        # get the profile samplings on surface
        sam_prof = surface_profiling_vert(vert_i, vert_norm_i, theta, r_step, m, intersect_mesh)

        # compute the 2D coordinates (x, y) for all profile points
        sample_x, sample_y = compute_profile_coord_x_y(np.array(sam_prof), vert[i], norm[i])

        profile_samples_x.append(sample_x)
        profile_samples_y.append(sample_y)

        print(i, 'is done')

    return np.array(profile_samples_x), np.array(profile_samples_y)


def surface_profiling_vert(vertex, vert_norm, theta, r, m, mesh):
    """
    Implement the profile sampling process for a given vertex.
    :param vertex: the target vertex (center vertex) [x, y, z]
    :param vert_norm: the vertex normal
    :param theta:
    :param r:
    :param m:
    :param mesh:
    :return:
    """

    profile_list = []  # record all the samples x and y

    # set initial rotation direction
    # randomly select R0
    dir_r0 = np.array([1, 1, 1]) - vertex

    # project the dir_R0 onto the tangent plane
    r0 = project_vector2tangent_plane(vert_norm, dir_r0)[0]

    for i in range(int(360 / theta)):

        # set the rotation directions
        rotate_angle = (i * theta) * 1.0 / 360 * 2 * np.pi
        rot_mat = get_rotate_matrix(vert_norm, rotate_angle)
        r_alpha = np.dot(r0, rot_mat)
        p_norm = np.cross(vert_norm, r_alpha)

        intersect_lines = trimesh.intersections.mesh_plane(mesh, p_norm, vertex)
        lines_index = np.unique(intersect_lines, axis=0, return_index=True)[1]
        ordered_lines = intersect_lines[lines_index]

        points_i, points_index, _ = select_points_orientation(ordered_lines, r_alpha, vertex)

        if len(points_index) != 0:
            samples_y = []
            points_sign = []
            length_sum = np.linalg.norm(points_i[0] - vertex)  # record the length of segment
            minued_lenth = 0
            count_i = 0
            exceed_index = 0  # record the i when the sample distance firstly exceed the maximum of intersection
            exceed_bool = True
            count_max = len(points_i)

            for j in range(m):
                sample_dist = (j + 1) * r
                if sample_dist <= length_sum and count_i == 0:
                    # the point is between center point and the closest one
                    point0 = vertex
                    points1 = points_i[0]
                    # alpha = sample_dist/np.linalg.norm(points1 - point0)
                elif sample_dist <= length_sum and count_i != 0:
                    point0 = points_i[count_i - 1]
                    points1 = points_i[count_i]
                    # length_sum += np.linalg.norm(points1 - point0)
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
                        sample_dist = (exceed_index + 1) * r
                        point0 = points_i[count_i - 1]
                        points1 = points_i[count_i]
                    else:
                        point0 = points_i[count_i - 1]
                        points1 = points_i[count_i]
                        length_sum += np.linalg.norm(points1 - point0)

                if np.linalg.norm(points1 - point0) == 0:
                    alpha = 0
                else:
                    alpha = (sample_dist - minued_lenth) / np.linalg.norm(points1 - point0)

                sample_point = (1 - alpha) * point0 + alpha * points1
                points_sign.append(sample_point)  # use to calculate the sign of sample y

        else:
            # the origin point is out of intersection points
            points_sign = list(np.zeros([m, 3]))

        profile_list.append(points_sign)

    return profile_list


def vert2poly_indices(vertex_array, poly_array):
    """
    Find vertex-polygon indices from the polygons array of vertices

    :param vertex_array:
    :param poly_array:
    :return:
    """

    vert_faces_arr = np.array([], dtype=int)
    for i in range(len(vertex_array)):
        face_i = np.where(poly_array == vertex_array[i])[0]
        vert_faces_arr = np.hstack((vert_faces_arr, face_i))

    return vert_faces_arr


def get_rotate_matrix(rot_axis, angle):
    """
    for a pair of rotation axis and angle, calculate the rotate matrix
    :param rot_axis: rotation axis
    :param angle: rotation angle
    :return: rotate matrix of [3, 3]
    """

    # normalize the rotate axis
    r_n = rot_axis / np.linalg.norm(rot_axis)
    rot_matrix = np.zeros((3, 3), dtype='float32')

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x = r_n[0]
    y = r_n[1]
    z = r_n[2]

    rot_matrix[0, 0] = cos_theta + (1 - cos_theta) * np.power(x, 2)
    rot_matrix[0, 1] = (1 - cos_theta) * x * y - sin_theta * z
    rot_matrix[0, 2] = (1 - cos_theta) * x * z + sin_theta * y

    rot_matrix[1, 0] = (1 - cos_theta) * y * x + sin_theta * z
    rot_matrix[1, 1] = cos_theta + (1 - cos_theta) * np.power(y, 2)
    rot_matrix[1, 2] = (1 - cos_theta) * y * z - sin_theta * x

    rot_matrix[2, 0] = (1 - cos_theta) * z * x - sin_theta * y
    rot_matrix[2, 1] = (1 - cos_theta) * z * y + sin_theta * x
    rot_matrix[2, 2] = cos_theta + (1 - cos_theta) * np.power(z, 2)

    return rot_matrix


def project_vector2tangent_plane(v_n, v_p):
    """
    calculate the projection vector of v_p onto tangent plane of v
    :param v_n: normal vector of v
    :type: ndarray
    :param v_p: vector projected
    :type: ndarray
    :return: v_t
    """
    unitev_n = v_n / np.linalg.norm(v_n)

    coeff_v_pn = np.dot(v_p, v_n) / np.linalg.norm(v_n)

    coeff = coeff_v_pn.reshape([coeff_v_pn.size, 1])

    v_pn = coeff * unitev_n

    v_t = v_p - np.array(v_pn)

    return v_t


def project_vector2vector(v_n, v_p):
    """
    calculate the projection vector of v_p onto v_n,
    v_pn = (v_p dot v_n) / |v_n| * unite vector of (v_n)
    :param v_n_: direction vector
    :type : ndarray
    :param v_p: vectors projected
    :type : ndarray
    :return:
    """
    unitev_n = v_n / np.linalg.norm(v_n)

    coeff_v_pn = np.dot(v_p, v_n) / np.linalg.norm(v_n)

    coeff = coeff_v_pn.reshape([coeff_v_pn.size, 1])

    v_pn = coeff * unitev_n

    return v_pn


def select_points_orientation(intersect_points, r_alpha, origin):
    """

    :param intersect_points:  NOTE: intersecting points of the same polygon are saved together
    :param r_alpha:
    :param origin:
    :return: r_points, points in the direction
             points_index, points indices
             line_index, the local indices of the segments that contain the intersecting points.
    """

    points_i = intersect_points.reshape(intersect_points.size // 3, 3)
    # points_i = np.array(list(points_i))  # align the array

    # find the center points
    origin_index = np.unique(np.where(points_i == origin)[0])

    if len(origin_index) == 0:
        # the intersection result exclude origin point
        r_points = []
        points_index = []
        line_index = []
        return r_points, points_index, line_index

    origin_points_i = origin_index // 2  # index in array 'intersect_points'

    # the relative position of origin in intersect_points[i]
    ori_points_index = origin_index % 2
    # the relative position of another point in intersect_points[i]
    another_points_index = (origin_index + 1) % 2

    target_point = points_i[origin_index][0]
    target_index = origin_index[0]

    # compute the dot between orientation Ra and two initial vector
    # respectively
    for i in range(len(origin_index)):
        point = intersect_points[origin_points_i[i]][another_points_index[i]]
        origin = intersect_points[origin_points_i[i]][ori_points_index[i]]

        dir_vec = point - origin  # direction vector
        dot_points = np.dot(dir_vec, r_alpha)

        if dot_points > 0:  # indicate this point in the same orientation
            target_point = point
            # get the index in array 'points_i'
            target_index = origin_points_i[i] * 2 + another_points_index[i]
            break

    # save the index of points that are in the orientation Ra
    points_index = [target_index]

    end_bool = False  # =True, if the search process arrives at the end of the orientation Ra
    while not end_bool:
        target_indices = np.unique(np.where(points_i == target_point)[0])
        if len(target_indices) == 1:
            # only one point in the array, means it is the end point
            break
        else:
            current_index = target_indices[np.where(
                target_indices != target_index)[0]][0]
            current_points_i = current_index // 2  # index in array 'intersect_points'

            # the relative position of another point in intersect_points[i]
            target_points_index = (current_index + 1) % 2
            # get the index in array 'points_i'
            target_index = current_points_i * 2 + target_points_index
            target_point = points_i[target_index]

        if target_index in points_index:
            break
        points_index.append(target_index)

    # ordered line index
    line_index = np.array(points_index) // 2

    r_points = points_i[points_index]
    return r_points, points_index, line_index


def compute_profile_coord_x_y(profile, origin, normal):
    """
    Calculate the 2D coordinates of the profiling points in their rotation planes
    :param profile:
    :param origin: center vertex of profiling
    :param normal:
    :return:
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

