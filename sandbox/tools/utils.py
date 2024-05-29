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


if __name__ == "__main__":
    mesh_path = "~/Documents/test_sulcal/01/mesh.gii"
    mesh = io.load_mesh(mesh_path)

    (shortest_path, field_tex) = differential_geometry.mesh_fiedler_length(mesh, dist_type="geodesic")
    start = shortest_path[0]
    end = shortest_path[-1]
    min_max = mesh.vertices[start, :] - mesh.vertices[end, :]
    dist = np.sqrt(np.sum(min_max * min_max, 0))
    print(dist)
    print(compute_dist(mesh, shortest_path))

    (shortest_path, field_tex) = differential_geometry.mesh_fiedler_length(mesh, dist_type="euclidian")
    print(shortest_path)