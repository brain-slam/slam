import os
import pickle
import numpy as np
from slam import io, geodesics, texture
import networkx as nx
import slam.watershed as swat


def extract_sulcal_graph(side, path_to_mesh, path_to_features,
                         path_to_output, path_to_mask=None):
    """
    Main Function that extracts the sulcal graph from a mesh and saves
    it in the given directory.
    Parameters
    ----------
    side
    path_to_mesh
    path_to_features
    path_to_output
    path_to_mask

    Returns
    -------

    """

    mesh = io.load_mesh(path_to_mesh)

    _, dpf, voronoi = swat.compute_mesh_features(mesh,
                                                 save=True,
                                                 outdir=path_to_features,
                                                 check_if_exist=True)
    thresh_dist, thresh_ridge, thresh_area = (
        swat.normalize_thresholds(mesh, voronoi,
                                  thresh_dist=20.0,
                                  thresh_ridge=1.5,
                                  thresh_area=50.0,
                                  side=side))
    if path_to_mask:
        mask = io.load_texture(path_to_mask).darray[0]
    else:
        mask = None

    basins, ridges, adjacency = (
        swat.watershed(mesh,
                       voronoi,
                       dpf,
                       thresh_dist,
                       thresh_ridge,
                       thresh_area, mask))
    g = get_sulcal_graph(adjacency,
                         basins,
                         ridges,
                         save=True,
                         outdir=path_to_output)
    get_textures_from_graph(g, mesh, save=True, outdir=path_to_output)
    return g


def get_sulcal_graph(adjacency, basins, ridges, save=True, outdir=None):
    """
    Function that creates a graph from the outputs of the watershed.

    Node attributes are:
    - pit_index: index of the pit
    - pit_depth: depth of the pit
    - basin_vertices: list of vertices in the basin
    - basin_area: area of the basin
    - basin_label: label of the basin

    Edge attributes are:
    - ridge_index: index of the ridge
    - ridge_depth: depth of the ridge point
    - ridge_length: number of vertices in the ridge

    Parameters
    ----------
    adjacency
    basins
    ridges
    save
    outdir

    Returns
    -------

    """

    ##############################################################
    # Initialize the graph using adjacency matrix
    ##############################################################

    # As the adjacency matrix concerns all created basins during watershed,
    # it still contains merged basins that should not appear in the graph.
    # Thus, we first remove rows and columns corresponding to unconnected
    # basins (only zeros inside)
    labels = list(basins.keys())
    adjacency = adjacency[labels, :][:, labels]
    # np.fill_diagonal(adjacency, 1.)  # ensure systematic self connection
    graph = nx.from_numpy_array(adjacency)  # , nodelist=basins)
    # nodelist not adapted to attribution of labels in plotly_visu.py

    ####################################################################
    # Set graph attributes
    ####################################################################

    # Add node attributes
    node_attributes = {}
    for i, (label, values) in enumerate(basins.items()):
        node_attributes[i] = basins[label]  # add all dictionary values
        node_attributes[i]['basin_label'] = label  # add label value
    nx.set_node_attributes(graph, node_attributes)

    # Add edge attributes
    edge_attributes = {}
    for pair, values in ridges.items():
        # Get new indices
        i = labels.index(pair[0])
        j = labels.index(pair[1])
        edge_attributes[i, j] = values  # add all dictionary values
        print(values)
    for ed in range(len(graph.edges)):
        print("Edge "+str(ed)+" attributes:",
              graph.edges[list(graph.edges)[ed]].keys())

    nx.set_edge_attributes(graph, edge_attributes)
    for ed in range(len(graph.edges)):
        print("Edge "+str(ed)+" attributes:",
              graph.edges[list(graph.edges)[ed]].keys())

    if save:
        if not outdir:
            outdir = ''
        save_graph(graph, outdir)

    return graph


def save_graph(graph, outdir):
    """
    Save sulcal pits graph in the given directory under the
    name "graph.gpickle"
    Parameters
    ----------
    graph
    outdir

    Returns
    -------

    """

    file_path = os.path.join(outdir, "graph.gpickle")
    with open(file_path, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    print("Graph saved in", file_path)
    return 0


def add_node_attribute_to_graph(graph, texture, name, save=True, outdir=None):
    """
    Add a node attribute to the graph using the value of the texture
    at pit positions
    Parameters
    ----------
    graph
    texture
    name
    save
    outdir

    Returns
    -------

    """

    if save and not outdir:
        outdir = ''

    node_values = {}
    for basin in graph.nodes:
        # Get pits indices
        pit = graph.nodes[basin]['pit_index']
        # Get the texture values for each pit
        node_values[basin] = texture[pit]

    # Add the attribute to the graph
    nx.set_node_attributes(graph, values=node_values, name=name)

    if save:
        save_graph(graph, outdir)

    return graph


def add_edge_attribute_to_graph(graph, texture, name, save=True, outdir=None):
    """
        Add an edge attribute to the graph using the value of the texture at
    ridge positions
    Parameters
    ----------
    graph
    texture
    name
    save
    outdir

    Returns
    -------

    """

    if save and not outdir:
        outdir = ''

    # Get the adjacency matrix with ridge positions
    adjacency = nx.to_numpy_array(graph, weight='ridge_index', dtype=np.int32)
    # Create and fill a new edge dictionary with the texture
    # values at ridge positions
    ridge_dict = {}
    for i, j in graph.edges:
        ridge_dict[(i, j)] = float(texture[adjacency[i][j]])
    # Add the attribute to the graph
    nx.set_edge_attributes(graph, ridge_dict, name=name)

    if save:
        save_graph(graph, outdir)

    return graph


def add_geodesic_distances_to_graph(graph, mesh, save=True, outdir=None):
    """
        Add the geodesic distances between ridge and pits to the corresponding
     ridge attributes in the graph:

    - geodesic_distance_btw_ridge_pit_i: geodesic distance between the
    ridge and the first pit
    - geodesic_distance_btw_ridge_pit_j: geodesic distance between the
    ridge and the second pit
    - geodesic_distance_btw_pits: geodesic distance between the two pits
    connected by the ridge (sum of previous values)
    Parameters
    ----------
    graph
    mesh
    save
    outdir

    Returns
    -------

    """
    if save and not outdir:
        outdir = ''

    # Create and fill a new edge dictionary with the geodesic distances
    geodistances = {}
    for i, j in graph.edges:
        ridge = graph.edges[(i, j)]['ridge_index']
        pit_i = graph.nodes[i]['pit_index']
        pit_j = graph.nodes[j]['pit_index']
        # Compute the geodesic distances between ridge and both pits
        gd_from_ridge = geodesics.compute_gdist(mesh, ridge)
        geodistances[(i, j)] = {}
        geodistances[(i, j)]['geodesic_distance_btw_ridge_pit_i'] = (
            float(gd_from_ridge[pit_i]))
        geodistances[(i, j)]['geodesic_distance_btw_ridge_pit_j'] = (
            float(gd_from_ridge[pit_j]))
        geodistances[(i, j)]['geodesic_distance_btw_pits'] = (
                float(gd_from_ridge[pit_i]) + float(gd_from_ridge[pit_j]))

    # Add the attribute to the graph
    nx.set_edge_attributes(graph, geodistances)

    if save:
        save_graph(graph, outdir)

    return graph


def add_mean_value_to_graph(graph, texture, name, save=True, outdir=None):
    """
        Add the mean value of the texture over the vertices of each basin
    to the graph node attributes
    Parameters
    ----------
    graph
    texture
    name
    save
    outdir

    Returns
    -------

    """
    if save and not outdir:
        outdir = ''

    average_values = {}
    for basin in graph.nodes:
        # Get the list of vertices
        vertices = graph.nodes[basin]['basin_vertices']
        # Compute the average value of texture over the vertices
        mean_value = np.mean(texture[vertices])
        average_values[basin] = mean_value

    # Add the attribute to the graph
    nx.set_node_attributes(graph, average_values, name=name)

    if save:
        save_graph(graph, outdir)

    return graph


def get_textures_from_graph(graph, mesh, save=True, outdir=None):
    """
    Function that returns the textures from a graph of sulcal pits
    Parameters
    ----------
    graph
    mesh
    save
    outdir

    Returns
    -------

    """
    if save and not outdir:
        outdir = ''

    vert = np.array(mesh.vertices)

    # texture of labels
    labels = np.full(len(vert), -1, dtype=np.int64)
    for b in graph.nodes:
        labels[graph.nodes[b]['basin_vertices']] = (
            graph.nodes)[b]['basin_label']
    tex_labels = texture.TextureND(darray=labels.flatten())
    if save:
        io.write_texture(tex_labels, os.path.join(outdir, "labels.gii"))

    # texture of pits
    atex_pits = np.zeros((len(vert), 1))
    pits_indices = list(nx.get_node_attributes(graph, 'pit_index').values())
    atex_pits[pits_indices] = 1
    tex_pits = texture.TextureND(darray=atex_pits.flatten())
    if save:
        io.write_texture(tex_pits, os.path.join(outdir,
                                                "pits_tex_from_graph.gii"))

    # texture of ridges
    atex_ridges = np.zeros((len(vert), 1))
    ridges_indices = (
        list(nx.get_edge_attributes(graph, 'ridge_index').values()))
    atex_ridges[ridges_indices] = 1
    tex_ridges = texture.TextureND(darray=atex_ridges.flatten())
    if save:
        io.write_texture(tex_ridges,
                         os.path.join(outdir, "rigdes_tex_from_graph.gii"))

    return tex_labels, tex_pits, tex_ridges
