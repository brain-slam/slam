
import numpy as np
from scipy import sparse
import trimesh
from trimesh import graph
import networkx as nx


def boundaries_intersection(boundary):
    bound_conn = []
    for bound_ind1 in range(len(boundary) - 1):
        for bound_ind2 in range(bound_ind1 + 1, len(boundary)):
            common = set(boundary[bound_ind1]).intersection(
                set(boundary[bound_ind2]))
            if common:
                bound_conn.append([bound_ind1, bound_ind2, list(common)])
    return bound_conn


def cat_boundary(bound1, bound2):
    # b_tmp=bound1.copy()
    # print(bound1)
    # print(bound2)
    # for x in bound2:
    #     bound1.append(x)
    # print(bound1)
    bound1.extend(bound2)
    return bound1


def cut_mesh(mesh, atex_in):
    """
    cut the mesh into submeshes following the values in the texture
    returns as many meshes as the number of different values in the texture
    the vertices on the border between two submeshes are duplicated into
    both submeshes
    :param mesh:
    :param atex_in:
    :return:
    """
    # see mesh.submesh(faces_sequence)
    atex = np.around(atex_in)
    atex2 = atex.copy()
    labels = np.unique(atex)
    labels = labels.tolist()
    labels.reverse()
    sub_meshes = list()
    sub_indexes = list()
    last_label = labels[-1]
    for label_ind in range(len(labels) - 1):
        (sub_mesh, sub_index) = simple_cut_mesh(mesh, atex, labels[label_ind])
        sub_meshes.append(sub_mesh)
        sub_indexes.append(sub_index.tolist())
        # boundary = texture_boundary(mesh, atex, labels[label_ind])
        boundary = texture_boundary_vertices(atex, labels[label_ind],
                                             mesh.vertex_neighbors)
        atex2[boundary] = last_label
    (sub_mesh, sub_index) = simple_cut_mesh(mesh, atex2, last_label)
    sub_meshes.append(sub_mesh)
    sub_indexes.append(sub_index.tolist())
    return sub_meshes, labels, sub_indexes


def edges_to_adjacency_matrix(mesh):
    """
    compute the adjacency matrix of a mesh
    adja(i,j) = 2 if the edge (i,j) is in two faces
    adja(i,j) = 1 means that i and j are on the boundary of the mesh
    adja(i,j) = 0 elsewhere
    :param mesh:
    :return:
    """
    adja = graph.edges_to_coo(mesh.edges,
                              data=np.ones(len(mesh.edges),
                                           dtype=np.int8))

    return sparse.triu(adja) + sparse.tril(adja).transpose()


def edges_to_boundary(edges):
    """
    build the boundary by traversing edges return list of connected components
    ORDERED ACCORDING TO THEIR LENGTH, i.e. THE FIRST THE SHORTEST
    complex boundary corresponds to multiple holes in the surface or bad shaped
    boundary
    :param edges:
    :return:
    """
    graph = nx.from_edgelist(edges)
    # import matplotlib.pyplot as plt
    # plt.subplot(111)
    # nx.draw(graph, with_labels=True, font_weight='bold')
    # plt.show()

    sub_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    sub_graphs_path = list()
    for sub_graph in sub_graphs:
        sub_graph_nodes = list(sub_graph.nodes)
        paths = sorted(list(nx.all_simple_paths(sub_graph, source=sub_graph_nodes[0], target=sub_graph_nodes[1])), key=len, reverse=True)
        sub_graphs_path.append(paths[0])

    return sorted(sub_graphs_path, key=len)


def ismember(ar1, ar2):
    (uni, u_inds) = np.unique(ar1, False, True)
    inds = np.in1d(uni, ar2)
    return np.reshape(inds[u_inds], ar1.shape)


def list_count(l):
    """
    count the occurrences of all items in a list and return a dictionary
    that is of the form {nb_occurence:list_item}, which is the opposite of
    standard implementation usually found on the web
    -----------------------------------------------------------------
    in python >= 2.7, collections may be used, see example below
    >> from collections import Counter
    >> z = ['blue', 'red', 'blue', 'yellow', 'blue', 'red']
    >> Counter(z)
    Counter({'blue': 3, 'red': 2, 'yellow': 1})
    :param l:
    :return:
    """
    return dict((l.count(it), it) for it in l)


def mesh_boundary(mesh):
    """
    compute borders of a mesh
    :param mesh:
    :return:
    """
    adja = edges_to_adjacency_matrix(mesh)
    r = sparse.extract.find(adja)
    li = r[0][np.where(r[2] == 1)]
    lj = r[1][np.where(r[2] == 1)]
    edges_boundary = np.vstack([li, lj]).T
    """
    # alternative implementation based on edges and grouping from trimesh
    # instead of adjacency matrix
    from trimesh import grouping
    groups = grouping.group_rows(mesh.edges_sorted, require_count=1)
    # vertex_boundary = np.unique(open_mesh.edges_sorted[groups])
    edges_boundary = mesh.edges_sorted[groups]
    """
    if li.size == 0:
        print('No holes in the surface !!!!')
        return np.array()
    else:
        return edges_to_boundary(edges_boundary)


def simple_cut_mesh(mesh, atex, val):
    poly = mesh.faces
    vert = mesh.vertices
    tex_val_indices = np.where(atex == val)[0]
    inds = ismember(poly, tex_val_indices)
    poly_set = poly[inds[:, 0] & inds[:, 1] & inds[:, 2], :]
    (uni, inds) = np.unique(poly_set, False, True)
    submesh = trimesh.Trimesh(faces=np.reshape(inds, poly_set.shape),
                              vertices=vert[uni, :], process=False)
    return submesh, tex_val_indices


def texture_boundary(mesh, atex, val):
    """
    compute indexes that are the boundary of a region defined by value
    in a texture
    :param mesh:
    :param atex:
    :param val:
    :return:
    """
    # see mesh.facets_boundary()
    tex_val_indices = np.where(atex == val)[0]
    if not tex_val_indices.size:
        print('no value ' + str(val) + ' in the input texture!!')
        return list()
    else:

        bound_verts = texture_boundary_vertices(atex, val,
                                                mesh.vertex_neighbors)
        # select the edges that are on the boundary in the polygons
        adja = edges_to_adjacency_matrix(mesh)
        r = sparse.extract.find(adja)
        inr0 = []
        inr1 = []
        for v in bound_verts:
            inr0.extend(np.where(r[0] == v)[0])
            inr1.extend(np.where(r[1] == v)[0])
        r[2][inr0] = r[2][inr0] + 1
        r[2][inr1] = r[2][inr1] + 1
        li = r[0][np.where(r[2] == 4)]
        lj = r[1][np.where(r[2] == 4)]
        edges_boundary = np.vstack([li, lj]).T
        return edges_to_boundary(edges_boundary)


def texture_boundary_vertices(atex, val, vertex_neighbors):
    """
    compute indices of vertices that are the boundary of a region defined by
    value in the texture, without any topological or ordering constraint
    :param atex:
    :param val:
    :param vertex_neighbors:
    :return:
    """
    tex_val_indices = np.where(atex == val)[0]
    if not tex_val_indices.size:
        print('no value ' + str(val) + ' in the input texture!!')
        return list()
    else:
        ####################################################################
        # print( 'the vertices on the boundary have the same texture value
        # (boundary inside the patch)'
        ####################################################################
        '''identify the vertices that are on the boundary,
        i.e that have at least one neigbor that has not the same value in the
        texture '''
        bound_verts = list()
        for i in tex_val_indices:
            ne_i = np.array(vertex_neighbors[i])
            # print( ne_i.size
            # print( np.intersect1d_nu(ne_i, tex_val_indices).size
            inters_size = np.intersect1d(ne_i, tex_val_indices).size
            if inters_size != ne_i.size:
                bound_verts.append(i)
        return bound_verts
