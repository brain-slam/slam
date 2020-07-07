import numpy as np


def angle(vec1, vec2):
    """
    Return the angle between two vectors
    :param vec1:
    :param vec2:
    :return:
    """
    return np.arccos(np.dot(vec1, vec2)/(np.sqrt(np.dot(vec1, vec1) *
                                                 np.dot(vec2, vec2))))


def dotprod(vec1, vec2):
    """
    Return the normalized dotprod between two vectors
    :param vec1:
    :param vec2:
    :return:
    """
    return np.dot(vec1, vec2)/(np.sqrt(np.dot(vec1, vec1)*np.dot(vec2, vec2)))


def compare_analytic_estimated_directions(analytic_directions,
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
        angular_error[i] = min(np.mod(angle0, np.pi/4),
                               np.mod(angle1, np.pi/4))
        dotprods[i, 0] = dotprod(analytic_directions[i, :],
                                 estimated_directions[i, :, 0])
        dotprods[i, 1] = dotprod(analytic_directions[i, :],
                                 estimated_directions[i, :, 1])
    return angular_error, dotprods
