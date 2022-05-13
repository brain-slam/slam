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
    ga = nx.from_edgelist([(e[0], e[1], {"length": L}) for e, L in zip(edges, length)])

    # run the shortest path query using length for edge weight
    path = nx.shortest_path(ga, source=start_idx, target=end_idx, weight="length")

    return path


def compute_gdist(mesh, vert_id):
    """
    This func computes the geodesic distance from one point to all vertices on
     mesh by using the gdist.compute_gdist().
    Actually, you can get the geo-distances that you want by changing the
    source and target vertices set.
    :param mesh: trimesh object
    :param vert_id: the point index
    :return:
    """
    vert = mesh.vertices
    poly = mesh.faces.astype(np.int32)

    source_index = np.array([vert_id], dtype=np.int32)
    target_index = np.linspace(0, len(vert) - 1, len(vert)).astype(np.int32)

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
