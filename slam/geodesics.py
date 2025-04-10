import numpy as np
import gdist
import networkx as nx


def shortest_path(mesh, start_idx, end_idx):
    # edges without duplication
    edges = mesh.edges_unique

    # the actual length of each unique edge
    length = mesh.edges_unique_length

    # create the graph with edge attributes for length
    # g = nx.Graph()
    # for edge, L in zip(edges, length):
    #     g.add_edge(*edge, length=L)

    # alternative method for weighted graph creation
    # you can also create the graph with from_edgelist and
    # a list comprehension, which is like 1.5x faster

    ga = nx.from_edgelist([(e[0], e[1], {"length": L})
                           for e, L in zip(edges, length)])

    # run the shortest path query using length for edge weight
    path = nx.shortest_path(
        ga,
        source=start_idx,
        target=end_idx,
        weight="length")

    return path


def compute_gdist(mesh, source_id, target_id=None):
    """
    This func computes the geodesic distance between a set
    of sources and targets on a mesh surface by using the
    gdist.compute_gdist().
    If no target_id are provided, all vertices of the mesh
    are considered as targets.

    :param mesh: (trimesh object) the mesh surface
    :param source_id: (list) the sources index
    :param target_id: (list) the targets index
    :return:
    """
    vert = mesh.vertices
    poly = mesh.faces.astype(np.int32)

    source_index = np.array([source_id], dtype=np.int32)
    if target_id:
        target_index = np.array([target_id], dtype=np.int32)
    else:
        target_index = np.linspace(0, len(vert) - 1,
                                   len(vert)).astype(np.int32)

    return gdist.compute_gdist(vert, poly, source_index, target_index)


def local_gdist_matrix(mesh, max_geodist):
    """
    For every vertex, get all the vertices within the maximum distance.
    NOTE: It will take some time to compute all the geo-distance from point to
     point.
    Details about the computing time and memory see in method
    gdist.local_gdist_matrix
    :param mesh:
    :param max_geodist:
    :return:
    """

    vert = mesh.vertices
    poly = mesh.faces.astype(np.int32)

    return gdist.local_gdist_matrix(vert, poly, max_geodist)
