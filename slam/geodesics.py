import numpy as np
import gdist
import networkx as nx

VERBOSE = False


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
    path = nx.shortest_path(ga,
                            source=start_idx,
                            target=end_idx,
                            weight="length")

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


def gdist_length(mesh, start_indices):
    """
    Computes the distance to a set of points with 
    1. Exact algorithm of gdist applied to each point of the set
    2. Take the mini of distance maps for each point of the mesh
    CAREFUL: takes 3 to 5 times more than dijkstra_length
    :param mesh: a trimesh object with n vertices
    :param start_indices: indices of the set of points 
    :return: length, a distance map, array of size (n,)
    """
    lengths = gdist_lengths(mesh, start_indices)
    length = np.min(lengths, axis=1)
    return length


def gdist_lengths(mesh, start_indices):
    """
    Intermediate step to compute all distances (Dijkstra)
    from start_indices to the other indices
    Return as much distance maps as start_indices
    edges without duplication
    :param mesh: a trimesh object with n vertices
    :param start_indices: indices of a set of points (size k)
    :return: lengths, array of size (n,k)
    """
    vert = mesh.vertices
    poly = mesh.faces.astype(np.int32)
    lengths = np.zeros((len(vert,), len(start_indices)))

    target_index = np.linspace(0, len(vert) - 1, len(vert)).astype(np.int32)

    for i, vert_id in enumerate(start_indices):
        source_index = np.array([vert_id], dtype=np.int32)
        lengths[:, i] = gdist.compute_gdist(vert, poly,
                                            source_index, target_index)

    return lengths


def dijkstra_length(mesh, start_indices):
    """
    Computes the distance to a set of points with 
    1. Dijkstra algorithm applied to each point of the set
    2. Take the mini of distance maps for each point of the mesh
    :param mesh: a trimesh object with n vertices
    :param start_indices: indices of the set of points 
    :return: length, a distance map, array of size (n,)
    """
    lengths, ga = dijkstra_lengths(mesh, start_indices)
    length = np.min(lengths, axis=1)
    return length


def dijkstra_lengths(mesh, start_indices):
    """
    Intermediate step to compute all distances (Dijkstra)
    from start_indices to the other indices
    Return as much distance maps as start_indices
    edges without duplication
    :param mesh: a trimesh object with n vertices
    :param start_indices: indices of a set of points (size k)
    :return: lengths, array of size (n,k)
    """

    mod = 1
    if len(start_indices) > 100:
        mod = 10

    edges = mesh.edges_unique

    # the actual length of each unique edge
    length = mesh.edges_unique_length
    print(length)

    # create the graph with edge attributes for length
    ga = nx.from_edgelist([(e[0], e[1], {"length": L})
                           for e, L in zip(edges, length)])
    length_dijk = np.zeros((len(mesh.vertices), len(start_indices)))
    for i, vert_id in enumerate(start_indices):
        dict_length = nx.single_source_dijkstra_path_length(ga,
                                                            vert_id,
                                                            weight="length")
        for key in dict_length.keys():
            length_dijk[key, i] = dict_length[key]
        if i % mod == 0 and VERBOSE:
            print(i)
    return length_dijk, ga
