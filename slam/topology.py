
import numpy as np
from scipy import sparse
import trimesh

np_ver = [1, 6]  # [ int(x) for x in np.__version__.split( '.' ) ]


def ismember(ar1, ar2):
    if np_ver < [1, 6]:
        (uni, u_inds) = np.unique1d(ar1, False,
                                    True)  # deprecated since numpy version>1.6
        inds = np.setmember1d(uni, ar2)
    else:
        (uni, u_inds) = np.unique(ar1, False, True)
        inds = np.in1d(uni, ar2)
    return np.reshape(inds[u_inds], ar1.shape)


def edges_to_adjacency_matrix(mesh):
    """
    compute the adjacency matrix of a mesh
    adja(i,j) = 2 if the edge (i,j) is in two faces
    adja(i,j) = 1 means that i and j are on the boundary of the mesh
    adja(i,j) = 0 elsewhere
    :param mesh:
    :return:
    """
    return mesh.edges_sparse.astype('int8')


def edges_to_boundary(li, lj):
    """
    build the boundary by traversing edges return list of connected components
    ORDERED ACCORDING TO THEIR LENGTH, i.e. THE FIRST THE SHORTEST
    complex boundary corresponds to multiple holes in the surface or bad shaped
    boundary
    :param li:
    :param lj:
    :return:
    """
    tag = np.zeros(li.size)
    boundary = [[]]
    bound_ind = 0
    curr_edge_i = 0
    boundary[bound_ind].extend([li[curr_edge_i], lj[curr_edge_i]])
    tag[curr_edge_i] = 1

    reverse = 1
    while np.where(tag == 1)[0].size != tag.size:
        p = boundary[bound_ind][-1]
        curr_edge_i = np.where((li == p) & (tag == 0))[0]
        if curr_edge_i.size == 0:
            curr_edge_j = np.where((lj == p) & (tag == 0))[0]
            if curr_edge_j.size == 0:
                if reverse:
                    boundary[bound_ind].reverse()
                    reverse = 0
                else:
                    bound_ind += 1
                    reverse = 1
                    new_first = np.where((tag == 0))[0][0]
                    boundary.append([li[new_first], lj[new_first]])
                    tag[new_first] = 1
            else:
                boundary[bound_ind].append(li[curr_edge_j[0]])
                tag[curr_edge_j[0]] = 1
        else:
            boundary[bound_ind].append(lj[curr_edge_i[0]])
            tag[curr_edge_i[0]] = 1
    # concatenate boundary pieces
    bound_conn = boundaries_intersection(boundary)
    while len(bound_conn) > 0:
        cat_bound = cat_boundary(boundary[bound_conn[0][0]],
                                 boundary[bound_conn[0][1]])
        boundary[bound_conn[0][0]] = cat_bound
        boundary.pop(bound_conn[0][1])
        bound_conn = boundaries_intersection(boundary)

    for bound in boundary:
        if bound[0] == bound[-1]:
            bound.pop()

    # cut complex boundaries
    for b_ind, bound in enumerate(boundary):
        occurence = list_count(bound)
        if max(occurence.keys()) > 1:
            print('complex boundary --> cut into simple parts')
            while max(occurence.keys()) > 1:
                # find the vertex that is taken more than one time
                ite = occurence[max(occurence.keys())]
                first_occ = bound.index(ite)
                sec_occ = first_occ + 1 + bound[first_occ + 1:].index(ite)
                # create a new boundary that corresponds to the loop
                boundary.append(bound[first_occ:sec_occ])
                # remove the loop from current boundary
                bound[first_occ:sec_occ] = []
                occurence = list_count(bound)
            boundary[b_ind] = bound

    # sort the boundaries the first the longest
    boundaries_len = [len(bound) for bound in boundary]
    inx = np.array(boundaries_len).argsort()
    sort_boundary = [boundary[i] for i in inx]
    return sort_boundary


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


def cat_boundary(bound1, bound2):
    # b_tmp=bound1.copy()
    # print(bound1)
    # print(bound2)
    # for x in bound2:
    #     bound1.append(x)
    # print(bound1)
    bound1.extend(bound2)
    return bound1


def boundaries_intersection(boundary):
    bound_conn = []
    for bound_ind1 in range(len(boundary) - 1):
        for bound_ind2 in range(bound_ind1 + 1, len(boundary)):
            common = set(boundary[bound_ind1]).intersection(
                set(boundary[bound_ind2]))
            if common:
                bound_conn.append([bound_ind1, bound_ind2, list(common)])
    return bound_conn


def edges_to_simple_boundary(li, lj):
    tag = np.zeros(li.size)
    boundary = {}
    bound_ind = 0
    curr_edge_i = 0
    boundary[bound_ind] = [li[curr_edge_i], lj[curr_edge_i]]
    tag[curr_edge_i] = 1

    reverse = 1
    while (np.where(tag == 1)[0].size != tag.size):
        p = boundary[bound_ind][-1]
        curr_edge_i = np.where((li == p) & (tag == 0))[0]
        if (curr_edge_i.size == 0):
            curr_edge_j = np.where((lj == p) & (tag == 0))[0]
            if (curr_edge_j.size == 0):
                if reverse:
                    boundary[bound_ind].reverse()
                    reverse = 0
                else:  # multiple boundaries in this mesh
                    if boundary[bound_ind][0] == boundary[bound_ind][-1]:
                        boundary[bound_ind].pop()
                    bound_ind += 1
                    reverse = 1
                    new_first = np.where((tag == 0))[0][0]
                    boundary[bound_ind] = [li[new_first], lj[new_first]]
                    tag[new_first] = 1
            else:
                boundary[bound_ind].append(li[curr_edge_j[0]])
                tag[curr_edge_j[0]] = 1
        else:
            boundary[bound_ind].append(lj[curr_edge_i[0]])
            tag[curr_edge_i[0]] = 1
    return boundary


def mesh_boundary(mesh):
    """
    compute borders of a mesh
    :param mesh:
    :return:
    """
    adja = edges_to_adjacency_matrix(mesh)
    adja_tri = sparse.triu(adja) + sparse.tril(adja).transpose()
    r = sparse.extract.find(adja_tri)
    li = r[0][np.where(r[2] == 1)]
    lj = r[1][np.where(r[2] == 1)]

    if li.size == 0:
        print('No holes in the surface !!!!')
        return np.array()
    else:
        return edges_to_boundary(li, lj)


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
            if np_ver < [1, 6]:
                inters_size = np.intersect1d_nu(ne_i, tex_val_indices).size
            else:
                inters_size = np.intersect1d(ne_i, tex_val_indices).size
            if inters_size != ne_i.size:
                bound_verts.append(i)
        return bound_verts


def texture_boundary(mesh, atex, val):
    """
    compute indexes that are the boundary of a region defined by value
    in a texture
    :param mesh:
    :param atex:
    :param val:
    :param neigh:
    :return:
    """
    # voir mesh.facets_boundary()
    tex_val_indices = np.where(atex == val)[0]
    if not tex_val_indices.size:
        print('no value ' + str(val) + ' in the input texture!!')
        return list()
    else:

        bound_verts = texture_boundary_vertices(atex, val,
                                                mesh.vertex_neighbors)
        # select the edges that are on the boundary in the polygons
        adja = edges_to_adjacency_matrix(mesh)
        adja_tri = sparse.triu(adja) + sparse.tril(adja).transpose()
        r = sparse.extract.find(adja_tri)
        inr0 = []
        inr1 = []
        for v in bound_verts:
            inr0.extend(np.where(r[0] == v)[0])
            inr1.extend(np.where(r[1] == v)[0])
        r[2][inr0] = r[2][inr0] + 1
        r[2][inr1] = r[2][inr1] + 1
        li = r[0][np.where(r[2] == 4)]
        lj = r[1][np.where(r[2] == 4)]

        return edges_to_boundary(li, lj)


def cut_mesh(mesh, atex):
    """
    cut a hole in a mesh at nodes defined by value in texture
    returns two meshes of hole and mesh-hole
    the hole border belongs to both meshes
    :param mesh:
    :param atex:
    :return:
    """
    # voir mesh.submesh(faces_sequence)
    atex2 = atex.copy()
    labels = np.around(np.unique(atex))
    labels = labels.tolist()
    labels.reverse()
    sub_meshes = list()
    sub_indexes = list()
    last_label = labels[-1]
    for label_ind in range(len(labels) - 1):
        (sub_mesh, sub_index) = sub_cut_mesh(mesh, atex, labels[label_ind])
        sub_meshes.append(sub_mesh)
        sub_indexes.append(sub_index.tolist())
        #boundary = texture_boundary(mesh, atex, labels[label_ind])
        boundary = texture_boundary_vertices(atex, labels[label_ind], mesh.vertex_neighbors)
        atex2[boundary] = last_label
    (sub_mesh, sub_index) = sub_cut_mesh(mesh, atex2, last_label)
    sub_meshes.append(sub_mesh)
    sub_indexes.append(sub_index.tolist())
    return sub_meshes, labels, sub_indexes


def sub_cut_mesh(mesh, atex, val):
    poly = mesh.faces
    vert = mesh.vertices
    tex_val_indices = np.where(atex == val)[0]
    inds = ismember(poly, tex_val_indices)
    poly_set = poly[inds[:, 0] & inds[:, 1] & inds[:, 2], :]
    #    print( tex_val_indices
    (uni, inds) = np.unique(poly_set, False, True)
    submesh = trimesh.Trimesh(faces=np.reshape(inds, poly_set.shape),
                              vertices=vert[uni, :], process=False)
    return submesh, tex_val_indices
