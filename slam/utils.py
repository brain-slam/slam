import numpy as np


def clamp(val, a, b):
    if val < a:
        return a
    if val > b:
        return b
    return val


def angle(vec1, vec2, abs=False):
    """
    Return the angle between two vectors
    :param vec1:
    :param vec2:
    :return:
    """
    dp = dotprod(vec1, vec2)
    if abs:
        dp = np.abs(dp)
    return np.arccos(clamp(dp, -1, 1))


def dotprod(vec1, vec2):
    """
    Return the normalized dotprod between two vectors
    :param vec1:
    :param vec2:
    :return:
    """
    return np.dot(vec1, vec2) / \
        (np.sqrt(np.dot(vec1, vec1) * np.dot(vec2, vec2)))


def compare_analytic_estimated_directions(analytic_directions,
                                          estimated_directions, abs=False):
    """
    Compare the analytic principal directions with a estimated directions
    :param analytic_directions: (n,3) array
    :param estimated_directions: (n,3) array with one direction
    :return: angular error distribution
    """
    n = analytic_directions.shape[0]
    angular_error = np.zeros((n,))
    dotprods = np.zeros((n,))
    for i in range(n):
        angle0 = angle(analytic_directions[i, :],
                       estimated_directions[i, :], abs)
        angular_error[i] = angle0
        dotprods[i] = dotprod(analytic_directions[i, :],
                              estimated_directions[i, :])

    return angular_error, dotprods


def compare_analytic_estimated_directions_min(analytic_directions,
                                              estimated_directions):
    """
    Compare the analytic principal directions with a estimated directions
    :param analytic_directions: (n,3) array
    :param estimated_directions: (n,3,2) array with the two directions
    that may be not necessarily ordered
    :return: angular error distribution
    """
    n = analytic_directions.shape[0]
    angular_error = np.zeros((n,))
    dotprods = np.zeros((n, 2))
    for i in range(n):
        angle0 = angle(analytic_directions[i, :],
                       estimated_directions[i, :, 0])
        angle1 = angle(analytic_directions[i, :],
                       estimated_directions[i, :, 1])
        angular_error[i] = min(np.mod(angle0, np.pi / 4),
                               np.mod(angle1, np.pi / 4))
        dotprods[i, 0] = dotprod(analytic_directions[i, :],
                                 estimated_directions[i, :, 0])
        dotprods[i, 1] = dotprod(analytic_directions[i, :],
                                 estimated_directions[i, :, 1])
    return angular_error, dotprods


def get_rotate_matrix(rot_axis, angle):
    """
    for a pair of rotation axis and angle, calculate the rotate matrix
    :param rot_axis: rotation axis
    :param angle: float, the rotation angle is a real number rather than degree
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
