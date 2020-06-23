import numpy as np
from scipy import sparse
import trimesh
from trimesh import graph
import networkx as nx


# groups = graph.connected_components(mesh.face_adjacency)


def boundary_angles(boundary, vertices_coord):
    """
    compute the angle between consecutive edges in a path
    :param boundary: list of vertex indices corresponding to the path
    :param vertices_coord: array (n, 3) of vertex coordinates
    :return:
    """
    # pp = coordinates of consecutive vertices, including last-to-first to
    # close the path
    pp = np.zeros((len(boundary), 3))
    pp[:-1, :] = \
        vertices_coord[boundary[:-1], :] - vertices_coord[boundary[1:], :]
    pp[-1, :] = \
        vertices_coord[boundary[-1], :] - vertices_coord[boundary[0], :]
    norm = np.sqrt(np.dot(pp ** 2, [1] * pp.shape[1]))
    # tile reciprocal of norm
    i_norm = norm ** -1
    tiled = np.tile(i_norm, (pp.shape[1], 1)).T
    # multiply by reciprocal of norm
    u_pp = pp * tiled
    # u_pp is the unit vectors corresponding to the edges
    # u_qq is a copy of u_pp with a shift in indices of 1, so that u_pp and
    # u_qq are unit vectors of two consecutive edges
    u_qq = pp.copy()
    u_qq[1:, :] = u_pp[:-1, :]
    u_qq[0, :] = u_pp[-1, :]
    # dot product and angle computation
    dots = np.dot(u_pp * u_qq, [1.0] * u_pp.shape[1])
    ang = np.pi - np.arccos(np.clip(dots, -1, 1))
    ang = ang * 180 / np.pi
    return ang, norm


def boundaries_intersection(boundaries):
    """
    compute the intersections inside a boundary
    :param boundaries: list of list of vertex indices corresponding to the path
    :return: list of common vertices between each tuple
    """
    bound_conn = []
    for bound_ind1 in range(len(boundaries) - 1):
        for bound_ind2 in range(bound_ind1 + 1, len(boundaries)):
            common = set(boundaries[bound_ind1]).intersection(
                set(boundaries[bound_ind2]))
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


def close_mesh(mesh, boundary_in=None):
    """
    see also trimesh.repair.fill_holes(mesh):
    This function implements the following paper with slight modifications.
    Zhao, W., Gao, S., & Lin, H. (2007). A robust hole-filling algorithm for
     triangular mesh. The Visual Computer, 23(12), 987–997.
     https://doi.org/10.1007/s00371-007-0167-y
    :param mesh:
    :param boundary_in:
    :return:
    """
    if boundary_in is None:
        boundary_in = mesh_boundary(mesh)
    if len(boundary_in) == 0:  # no hole in the mesh, nothing to do!
        nb_verts_added = []
        return mesh, nb_verts_added

    """ copy mesh arrays to avoid caching from trimesh"""
    vertices = np.asanyarray(mesh.vertices)
    faces = np.asanyarray(mesh.faces)
    face_normals = np.asanyarray(mesh.face_normals)

    nb_verts_added = 0
    for bound_i in boundary_in:
        bound = bound_i.copy()
        bound.reverse()
        """ closing the hole with front propagation """
        f_b = ismember(faces, bound)
        boundIndsF = np.where(f_b[:, 0] | f_b[:, 1] | f_b[:, 1])[0]
        [bound_ang, edge_length] = boundary_angles(bound, vertices)
        r_min = min(edge_length)
        peaks = list()
        skip_angle_test = False
        while len(bound) > 2:
            nb_verts = 0
            admissible = True
            l_verts = vertices.shape[0]
            lb = len(bound) - 1
            # print(lb)
            # l_faces = faces.shape[0]
            # print(l_verts, l_faces)
            # print(bound_ang)
            bound_ang_m = min(bound_ang)
            ind_m = np.argmin(bound_ang)
            if bound_ang_m == np.inf:
                # only problematic angles around the boundary!
                # print('warning :: forced closing')
                [bound_ang, edge_length] = boundary_angles(bound, vertices)
                ind_m = np.argmin(bound_ang)
                bound_ang_m = 0
                skip_angle_test = True

            """ creating faces """
            if bound_ang_m < 75:
                if ind_m == 0:
                    add_faces = [bound[ind_m], bound[ind_m + 1], bound[lb]]
                elif ind_m == lb:
                    add_faces = [bound[ind_m], bound[0], bound[ind_m - 1]]
                else:
                    add_faces = [bound[ind_m], bound[ind_m + 1],
                                 bound[ind_m - 1]]
                nb_faces = 1

                K1 = ismember(faces[boundIndsF, :], add_faces[0])
                K2 = ismember(faces[boundIndsF, :], add_faces[1])
                K3 = ismember(faces[boundIndsF, :], add_faces[2])

                Ks = K1 | K2 | K3
                indsF = Ks[:, 0] & Ks[:, 1] & Ks[:, 2]
                if any(indsF):  # this face already exists!!!!
                    admissible = False
                    bound_ang[ind_m] = np.inf
                    peaks.append(bound[ind_m])
                else:
                    if len(bound) < 4:
                        add_faces = bound
                    else:
                        Ks = K2 | K3
                        indsF = np.sum(Ks, 1)
                        if max(indsF) > 1:  # the new edge already exists!
                            admissible = False
                            bound_ang[ind_m] = np.inf
            else:  # if bound_ang_m<135
                if ind_m == 0:
                    tri_verts = [bound[ind_m], bound[ind_m + 1], bound[lb]]
                    add_faces = [[bound[ind_m], bound[ind_m + 1], l_verts],
                                 [bound[ind_m], l_verts, bound[lb]]]
                elif ind_m == lb:
                    tri_verts = [bound[ind_m], bound[0], bound[ind_m - 1]]
                    add_faces = [[bound[ind_m], bound[0], l_verts],
                                 [bound[ind_m], l_verts, bound[ind_m - 1]]]
                else:
                    tri_verts = [bound[ind_m], bound[ind_m + 1],
                                 bound[ind_m - 1]]
                    add_faces = [[bound[ind_m], bound[ind_m + 1], l_verts],
                                 [bound[ind_m], l_verts, bound[ind_m - 1]]]
                nb_faces = 2
                nb_verts = 1
                if bound_ang_m < 135:
                    [add_verts, r] = create_vertex(vertices[tri_verts, :])
                else:  # bound_ang>135
                    # print('big angle!')
                    [add_verts, r] = create_vertex(vertices[tri_verts, :],
                                                   r=r_min)
                comp_faces = list(set(np.unique(
                    faces[boundIndsF, :])).difference(set(tri_verts)))
                comp_verts = vertices[comp_faces, :]
                dist_comp_verts = \
                    comp_verts - np.tile(add_verts, (comp_verts.shape[0], 1))
                """
                testing if the created vertex is not too close to another
                existing vertex
                """
                if min(np.sum(dist_comp_verts ** 2, 1) - r_min ** 2) < 0:
                    admissible = False
                    bound_ang[ind_m] = 0  # => close with one single face

            if admissible:
                if skip_angle_test:
                    teta = np.inf
                else:
                    """
                    test angle between normals,
                    if length(add_faces(:,1))>1 test only the first face as
                    they are in the same plane
                    """
                    norm_add_faces = np.zeros(3)
                    if nb_faces > 1:
                        pp = vertices[add_faces[0][1], :] \
                            - vertices[add_faces[0][0], :]
                        qq = add_verts - vertices[add_faces[0][0], :]
                    else:
                        pp = \
                            vertices[add_faces[1], :]
                        - vertices[add_faces[0], :]
                        qq = \
                            vertices[add_faces[2], :]
                        - vertices[add_faces[0], :]
                    norm_add_faces[0] = pp[1] * qq[2] - pp[2] * qq[1]
                    norm_add_faces[1] = pp[2] * qq[0] - pp[0] * qq[2]
                    norm_add_faces[2] = pp[0] * qq[1] - pp[1] * qq[0]
                    norm = np.sqrt(np.sum(norm_add_faces ** 2))
                    norm_add_faces = norm_add_faces / norm

                    f_b = ismember(faces[boundIndsF, :], bound[ind_m])
                    boundsVertIndsF = \
                        boundIndsF[f_b[:, 0] | f_b[:, 1] | f_b[:, 2]]
                    boundary_normalf = face_normals[boundsVertIndsF, :]
                    teta = max(np.dot(boundary_normalf, norm_add_faces))
                """
                teta-np.cos(np.pi/2) <=> acos()>pi/2 and np.cos(np.pi/2)=0
                """
                if teta < 0:
                    peaks.append(bound[ind_m])
                    bound_ang[ind_m] = np.inf
                else:
                    skip_angle_test = False
                    if nb_verts > 0:
                        vertices = np.vstack((vertices, add_verts))
                        nb_verts_added = nb_verts_added + 1
                    faces = np.vstack((faces, add_faces))
                    # print(np.array(add_faces))
                    # print(vertices[np.array(add_faces)])
                    add_normals, valid = trimesh.triangles.normals(
                        vertices[np.array(add_faces)])
                    face_normals = np.vstack((face_normals, add_normals))
                    """ update of the boundary,  boundIndsF and bound_ang"""
                    if nb_faces < 2:
                        bound.pop(ind_m)
                    else:  # nb_faces==2
                        bound[ind_m] = l_verts
                    f_b = ismember(faces, bound)
                    boundIndsF = np.where(f_b[:, 0] | f_b[:, 1] | f_b[:, 2])[0]
                    [bound_ang, edge_length] = boundary_angles(bound, vertices)

    """ barycentric smoothing of the new vertices """
    if peaks is not None:
        smoothed_peak = local_barycentric_smoothing(faces, vertices, peaks)
        vertices[peaks, :] = smoothed_peak
    # creating the output mesh, note that face normals are also recomputed here
    closed_mesh = trimesh.Trimesh(faces=faces,
                                  vertices=vertices,
                                  metadata=mesh.metadata,
                                  process=False)

    return closed_mesh, nb_verts_added


def local_barycentric_smoothing(faces, vertices, vertices_to_smooth):
    """
    local smoothing of the coordinates of vertices_to_smooth
    the smoothed coordinates of each vertex in vertices_to_smooth in vertices
    corrrespond to the barycenter of its neigbhors
    :param faces:
    :param vertices:
    :param vertices_to_smooth:
    :return:
    """
    smoothed_vertices = np.zeros((len(vertices_to_smooth), 3))
    for p, peak in enumerate(vertices_to_smooth):
        f_b = ismember(faces, peak)
        neigs = np.unique(faces[f_b[:, 0] | f_b[:, 1] | f_b[:, 2], :])
        smoothed_vertices[p, :] = np.mean(vertices[neigs, :], 0)
    return smoothed_vertices


def create_vertex(three_vertices, r=None):
    """
    create a vertex along the bisector of the angle formed by three vertices
    :param three_vertices: array(3,3) coordinates of the three vertices
    :param r: distance to the new vertex on the bisector
    :return:
    """
    # the two vectors formed by the three vertices
    pp = three_vertices[1, :] - three_vertices[0, :]
    qq = three_vertices[2, :] - three_vertices[0, :]
    # their norm
    n_pp = np.sqrt(np.sum(pp ** 2))
    n_qq = np.sqrt(np.sum(qq ** 2))
    # the two vectors normalized
    u_pp = pp / n_pp
    u_qq = qq / n_qq

    if r is None:
        r = min(n_pp, n_qq)
    # compute the bisector direction
    bisec = u_qq + u_pp
    """
    ensure the distance between the new vertex and three_vertices[0,:]
    is equal to r
    """
    k = np.sqrt(r ** 2 * np.sum(bisec ** 2)) / np.sum(bisec ** 2)
    new_vert = k * bisec + three_vertices[0, :]
    return new_vert, r


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
        (sub_mesh, sub_index) = sub_cut_mesh(mesh, atex, labels[label_ind])
        sub_meshes.append(sub_mesh)
        sub_indexes.append(sub_index.tolist())
        # boundary = texture_boundary(mesh, atex, labels[label_ind])
        boundary = texture_boundary_vertices(atex, labels[label_ind],
                                             mesh.vertex_neighbors)
        atex2[boundary] = last_label
    (sub_mesh, sub_index) = sub_cut_mesh(mesh, atex2, last_label)
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


def edges_to_boundary(edges_bound, mesh_edges):
    """
    build the boundary by traversing edges return list of connected components
    ORDERED ACCORDING TO THEIR LENGTH, i.e. THE FIRST THE SHORTEST
    complex boundary corresponds to multiple holes in the surface or bad shaped
    boundary
    In each boundary, the output is a list of vertex indices that respects
    the order of mesh edges i.e. travelling direction along the edges of the
    boundary
    is the same as in edges of the mesh
    :param edges_bound: edges from the mesh that constitue the boundary but are
    not ordered properly
    :param mesh_edges: edges of the mesh that will serve to define the
    travelling direction of the output boundary path
    :return: list of boundaries that are themself lists of vertices
    """
    graph = nx.from_edgelist(edges_bound)

    # import matplotlib.pyplot as plt
    # plt.subplot(111)
    # nx.draw(graph, with_labels=True, font_weight='bold')
    # plt.show()
    def search_edge(ed, edges):
        pos = list()
        for ind, e in enumerate(edges):
            if e[0] == ed[0] and e[1] == ed[1]:
                pos.append(ind)
        return pos

    sub_graphs = [graph.subgraph(c).copy() for c in
                  nx.connected_components(graph)]
    sub_graphs_path = list()
    for sub_graph in sub_graphs:
        # print(len(sub_graph.nodes))
        # plt.subplot(111)
        # nx.draw(sub_graph, with_labels=True, font_weight='bold')
        # plt.show()
        sub_graph_nodes = list(sub_graph.nodes)
        paths = sorted(list(
            nx.shortest_simple_paths(
                sub_graph, source=sub_graph_nodes[0],
                target=sub_graph_nodes[1])), key=len,
            reverse=True)
        longest_path = paths[0]
        if len(longest_path) < len(sub_graph.nodes):
            longest_path = paths[0][1:]
            shortest_path = paths[-1]
            shortest_path.reverse()
            longest_path.extend(shortest_path[1:])
        # print(len(longest_path))
        # print(len(paths))
        # print('---')
        # for p in paths:
        #     print('-'+str(len(p)))
        first_edge = longest_path[0:3]
        pos = search_edge(first_edge, mesh_edges)
        if len(pos) == 0:
            longest_path.reverse()
        sub_graphs_path.append(longest_path)

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
        return np.array([])
    else:
        return edges_to_boundary(edges_boundary, mesh.edges)


def sub_cut_mesh(mesh, atex, val):
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
        return edges_to_boundary(edges_boundary, mesh.edges)


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
