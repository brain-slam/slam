import numpy as np
from slam import io
import time
from slam import differential_geometry


def compute_dist(mesh, shortest_path):
    """
    Function that compute the euclidian distance for the shortest between 2 vertices
    Returns
    -------
    object
    """
    vertices = mesh.vertices[shortest_path]

    # Calcul des différences entre les points successifs
    differences = np.diff(vertices, axis=0)

    # Calcul de la distance euclidienne pour chaque segment
    distances = np.linalg.norm(differences, axis=1)

    # Somme des distances pour obtenir la longueur totale
    length = np.sum(distances)

    return length
