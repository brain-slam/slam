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