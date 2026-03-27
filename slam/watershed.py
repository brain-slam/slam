import numpy as np
from slam import (geodesics,
                  curvature, vertex_voronoi,
                  sulcal_depth, utils)


def compute_mesh_features(mesh):
    """
    Function that computes the mean curvature, the depth potential function
    and the voronoi areas of a mesh.
    Returns numpy arrays.
    Parameters
    ----------
    mesh

    Returns
    -------

    """
    # Compute vertex mean curvature
    print("\n\tComputing the curvature\n")
    PrincipalCurvatures, _, _ = curvature.curvatures_and_derivatives(mesh)
    mean_curv = (
            0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :]))
    filt_mean_curv = \
        (utils.z_score_filtering(np.array([mean_curv]), z_thresh=3))
    mean_curvature = filt_mean_curv[0]

    # Compute vertex depth potential function
    print("\n\tComputing the DPF\n")
    # returns a list of dpf for different alpha values
    dpf, lc = sulcal_depth.dpf_star(mesh,
                                    alphas=[500],
                                    curvature=mean_curvature)
    dpf = dpf[0]

    # Compute vertex voronoi areas
    print("\n\tComputing Voronoi's vertex\n")
    voronoi = vertex_voronoi.vertex_voronoi(mesh)

    return mean_curvature, dpf, voronoi


def normalize_thresholds(voronoi,
                         thresh_dist=20.0,
                         thresh_ridge=1.5,
                         thresh_area=50.0,
                         side="left"):
    """
    Function that normalizes the thresholds for the watershed algorithm.
    Threshold on basin's area is normalized by the surface area of the mesh.
    Threshold on distance between pits is normalized by the square root of
    the surface area.

    :args: voronoi(numpy array): voronoi areas of the mesh
    :args: thresh_dist(float): distance threshold for the watershed algorithm
    :args: thresh_ridge(float): ridge threshold for the watershed algorithm
    :args: thresh_area(float): area threshold for the watershed algorithm
    :args: side: side of the brain (left or right)
    :return: normalized thresholds (thresh_dist, thresh_ridge, thresh_area)

    Parameters
    ----------
    voronoi
    thresh_dist
    thresh_ridge
    thresh_area
    side

    Returns
    -------
    Normalized watershed parameters
    """

    # normalization
    print("\n\tComputing the surface area\n")
    mesh_area = np.sum(voronoi)
    mesh_area_sqrt = np.sqrt(mesh_area)

    # Set group average values
    # from OASIS database
    # group_average_fiedler_length = 238.25 if side == "right" else 235.95
    # group_average_surface_area = 91433.68 if side == "right" else 91369.33
    # from HCP database
    group_average_surface_area = 92161.30 if side == "right" else 91530.63
    group_average_surface_area_sqrt = 303.20 if side == "right" else 302.16

    # Normalize the thresholds regarding brain size
    thresh_dist *= mesh_area_sqrt / group_average_surface_area_sqrt
    thresh_area *= mesh_area / group_average_surface_area

    # Stay at input precision level
    thresh_dist = round(thresh_dist, 1)
    thresh_area = round(thresh_area, 1)

    return thresh_dist, thresh_ridge, thresh_area


def watershed(mesh, voronoi, dpf_star, thresh_dist_perc, thresh_ridge,
              thresh_area_perc, mask=None):
    """
    Sulcal segmentation and sulcal pits extraction using watershed by
    flooding.

    Reference paper is:
    G. Auzias, L.Brun, C. Deruelle, O. Coulon. 2015. Deep sulcal
    landmarks: algorithmic and conceptual improvements in the definition
     and extraction of sulcal pits. NeuroImage

    A related approach can be found in:
    Rettmann ME, Han X, Xu C, Prince JL. 2002. Automated
    sulcal segmentation using watersheds on the cortical
    surface. Neuroimage. 15:329-344.

    Parameters
    ----------
    mesh : white matter triangular mesh of subject (trimesh object)
    voronoi : voronoi area for each vertex (numpy array)
    dpf_star : depth measure for each vertex. Deepest points should have
     smaller values (numpy array)
    thresh_dist_perc : threshold on the distance between pits as a
        percentage of the characteristic length of the input mesh
    thresh_ridge : threshold on the ridge height
    thresh_area_perc : threshold on the basin area as a percentage of
        the surface area of the input mesh
    mask : binary array for cingular pole exclusion (with ones in the
    region to exclude)

    Returns
    -------
    basins : dictionary with basin properties
        basins[label] = {}
            'pit_index': vertex index of the sulcal pit
            'pit_depth': depth of the sulcal pit
            'basin_vertices': list of vertices in the basin
            'basin_area': area of the basin
    ridges : dict with properties of the ridge between basin i and basin j
        ridges[(i,j)] = {}
            'ridge_index': vertex index of the ridge point
            'ridge_depth': depth of the ridge point
            'ridge_depth_diff_min': depth difference between ridge point and
            shallowest pit
            'ridge_depth_diff_max': depth difference between ridge point and
            deepest pit
            'ridge_length': number of vertices along the frontier
            between basins
    adjacency : asymmetric adjacency matrix of the basins
        np.triu(adjacency)[i,j] = 1 if basin i and j are adjacent, 0 otherwise
    """

    # ====================================================================
    # NORMALIZE THE THRESHOLD VALUE FOR DISTANCE WRT CHARACTERISTIC LENGTH
    # compute the 1D characteristic length
    # as the cubic root of the volume of the convex hull of the input mesh
    hull = mesh.convex_hull
    vol_hull = hull.volume
    charac_length = np.power(vol_hull, 1/3)
    thresh_dist = thresh_dist_perc * charac_length / 100
    # =====================================================================
    # NORMALIZE THE THRESHOLD VALUE FOR AREA WRT THE AREA OF THE INPUT MESH
    thresh_area = thresh_area_perc * mesh.area / 100

    print('Computing watershed by flooding...')
    print('Thresholds provided:')
    print('-Distance between 2 pits:',
          thresh_dist_perc, '% <=>', thresh_dist, 'mm')
    print('-Ridge height:', thresh_ridge)
    print('-Basin area:', thresh_area_perc, '% <=>', thresh_area, 'mm2')

    # Initialize output tables and dictionaries
    n_vertices = mesh.vertices.shape[0]  # vertices coordinates (n, 3) float
    vert_idx = np.arange(n_vertices)  # vertex index
    vert_depth = dpf_star.reshape(n_vertices)
    vert_area = voronoi.reshape(n_vertices)
    # vertex labels (initialized to -1)
    vert_label = np.full(n_vertices, -1, dtype=np.int64)
    vert_neigh = mesh.vertex_neighbors  # list of lists of neighbors
    basins = {}
    adjacency = np.zeros((1, 1), dtype=np.int64)

    # Apply exclusion mask
    # All nodes included in the exclusion mask are not taken into
    # account in the watershed process
    if mask is None:
        mask = np.zeros(dpf_star.shape)
    mask_indices = np.where(mask == 1)[0]
    nodes = np.vstack((vert_idx, vert_depth)).T
    nodes = np.delete(nodes, mask_indices, axis=0)

    # Sorting step
    # All nodes of the mesh are sorted by their depth
    # (deepest nodes = smallest values first)
    nodes_sorted_by_depth = nodes[nodes[:, 1].argsort()]
    idx = int(nodes_sorted_by_depth[0, 0])
    # Initialize of the first basin with first pit: deepest node
    # the label (and thus the key) of this basin is 0
    new_label = 0
    vert_label[idx] = new_label
    basins[new_label] = {}
    basins[new_label]['pit_index'] = idx
    basins[new_label]['basin_vertices'] = [idx]
    # print("new basin label: ", new_label)

    for node in nodes_sorted_by_depth[1:]:

        idx = int(node[0])
        # indices of neighbors
        neigh_idx = np.array(vert_neigh[idx]).astype(int)
        # labels of neighbors
        neigh_labels = vert_label[neigh_idx]
        # Keep only labelled neighbors
        neigh_idx = neigh_idx[:, None][neigh_labels != -1]
        neigh_labels = neigh_labels[neigh_labels != -1]
        NL = np.unique(neigh_labels)

        #############################################################
        # Case 1: all of its neighbors are unlabeled. Then, this node
        # corresponds to the deepest point of a new catchment basin.
        #############################################################
        if len(NL) == 0:
            new_label += 1  # max(basins.keys()) + 1
            # print("new basin label: ", new_label)
            # Update vertex
            vert_label[idx] = new_label
            # Update basins
            basins[new_label] = {}
            basins[new_label]['pit_index'] = idx
            basins[new_label]['basin_vertices'] = [idx]
            # Update adjacency matrix: increment size by 1
            adjacency = np.pad(adjacency,
                               ((0, 1), (0, 1)),
                               mode='constant',
                               constant_values=0)

        #############################################################
        # Case 2: the node is the neighbor of only one catchment basin.
        # Then, this node is assigned to the corresponding basin.
        #############################################################
        elif len(NL) == 1:
            # Update vertex
            vert_label[idx] = NL[0]
            # Update basin
            basins[NL[0]]['basin_vertices'].append(idx)

        ##############################################################
        # Case 3: the node is the neighbor of two or more catchment basins.
        # Then, this node is a ridge point where each pair of basins join.
        # It is assigned to the basin represented by the deepest neighbor
        # vertex, or the lowest label if same depth.
        # Then the conditions for merging two basins are tested (distance
        # between the two pits and ridge height).
        ################################################################
        else:
            # deepest neighbor
            idx_max_depth = np.argmin(vert_depth[neigh_idx])
            lab = np.min(neigh_labels[idx_max_depth])  # lowest label
            # Update vertex
            vert_label[idx] = lab
            # Update basin
            basins[lab]['basin_vertices'].append(idx)

            # MERGING between pairs of neighbor catchment basins
            # The merging condition is questioned between the intersection
            # label (lab) and all other neighbors.
            # The merging condition is only questioned if the basins
            # have never met yet (no existing ridge point)

            # NL.sort()  # NL is already sorted when using np.unique()
            NL = np.delete(NL, NL == lab)
            for neighbor_lab in NL:
                # Check that basins have not already been merged in the loop
                if (neighbor_lab in np.unique(vert_label)
                        and lab in np.unique(vert_label)):
                    # Forcing label_i < label_j so that
                    # pit(i) is always deeper than pit(j)
                    label_i = min(lab, neighbor_lab)
                    label_j = max(lab, neighbor_lab)
                    if adjacency[label_i, label_j] == 0:  # first meet

                        merging = 0

                        # Compute ridge height
                        ridge_height = (
                            abs(vert_depth[basins[label_j]['pit_index']]
                                - node[1]))

                        # # debug: ensure that
                        # # ridge_height_label_i < ridge_height_label_j
                        # ridge_height_cus = (
                        #     abs(vert_depth[basins[label_i]['pit_index']]
                        #         - node[1]))
                        # print("ridge_height=", ridge_height)
                        # print("ridge_height_cus=", ridge_height_cus)
                        # print("ridge_height<ridge_height_cus :: ",
                        # ridge_height < ridge_height_cus)
                        # print(ridge_height, thresh_ridge)
                        if ridge_height < thresh_ridge:
                            # print("compute gdist")
                            # Compute distance between pits
                            v = geodesics.compute_gdist(
                                mesh,
                                basins[label_i]['pit_index'],
                                basins[label_j]['pit_index'])
                            if v < thresh_dist:
                                # print('merging of',label_j,'into',label_i)
                                merging = 1
                                # Update vertex
                                vert_label[
                                    np.where(vert_label == label_j)[0]]\
                                    = label_i
                                # Update basin
                                basins[label_i]['basin_vertices'].extend(
                                    basins[label_j]['basin_vertices'])
                                del basins[label_j]
                                # Update adjacency
                                # Add neighboring basins of j to i
                                adjacency[label_i, :] = (
                                        adjacency[label_i, :] |
                                        adjacency[label_j, :])
                                adjacency[:, label_i] = (
                                        adjacency[:, label_i] |
                                        adjacency[:, label_j])
                                # clean out adjacency of j
                                adjacency[label_j, :] = 0
                                adjacency[:, label_j] = 0

                                if label_j == lab:
                                    # lab disappears and is replaced by the
                                    # parent label which becomes adjacent
                                    # to other neighbors.
                                    # Continue the loop to test merge
                                    # with this new lab.
                                    lab = neighbor_lab

                        # Create a new entry in the ridge dictionary
                        if not merging:
                            adjacency[label_i, label_j] = 1
                            adjacency[label_j, label_i] = 1

    # print('Number of basins before filtering on basin area:', len(basins))
    #################################################################
    # FILTERING OF SMALL BASINS
    # Last merging step on area criterion after the watershed process
    # Each basin which area is less than thresh_area is merged into the
    # neighbor basin it shares the largest border with.
    ##################################################################
    # Select basins with area < thresh_area
    basins2merge = []
    for label in basins:
        basins[label]['pit_depth'] = vert_depth[basins[label]['pit_index']]
        basins[label]['basin_area'] = (
            np.sum(vert_area[basins[label]['basin_vertices']]))
        if basins[label]['basin_area'] <= thresh_area:
            basins2merge.append([label, basins[label]['basin_area']])
    basins2merge = np.array(basins2merge)

    # print('nb of basins to remove:', basins2merge.shape[0])
    if basins2merge.shape[0] > 0:
        # print('Filtering of small basins...')
        # Sort by area
        basins2merge = basins2merge[basins2merge[:, 1].argsort()]
        # Filtering
        basin_index = 0
        while basin_index != len(basins2merge):
            basin, area = basins2merge[basin_index]
            basin = int(basin)
            # Find neighboring basins
            neigh_basins = np.where(adjacency[basin, :] != 0)[0]
            if basin in neigh_basins:
                neigh_basins = np.delete(neigh_basins, neigh_basins == basin)
            # print(neigh_basins)

            # Find parent_basin as the one sharing the largest border
            # so the higher ridges_length
            ridges_length = []
            for n_b in neigh_basins:
                # List all ridge vertices
                ridges_vertices = []
                for v in basins[basin]['basin_vertices']:
                    neighbors_vertices = np.array(vert_neigh[v])
                    ridges_vertices.extend(
                        neighbors_vertices[
                            vert_label[neighbors_vertices] == n_b])
                # Get ridge length with each neighbor
                ridges_length.append(len(ridges_vertices))
            parent_basin = neigh_basins[np.argmax(ridges_length)]
            # print('merging basin', basin, 'into', int(parent_basin))

            # Update vertex label, that is used to find neigbhoring basins
            vert_label[np.where(vert_label == basin)[0]] = parent_basin
            # Update basin
            basins[parent_basin]['basin_vertices'].extend(
                basins[basin]['basin_vertices'])
            basins[parent_basin]['basin_area'] += area
            del basins[basin]
            # Update adjacency
            # Add neighboring basins of j to i
            adjacency[parent_basin, :] = (
                    adjacency[parent_basin, :] |
                    adjacency[basin, :])
            adjacency[:, parent_basin] = (
                    adjacency[:, parent_basin] |
                    adjacency[:, basin])
            adjacency[basin, :] = 0
            adjacency[:, basin] = 0

            # basins2merge must be updated:
            # if parent_basin was in the list of basins2merge,
            # its area has increased and the updated area of
            # parent_basin can now be larger than the threshold
            # In this case, we remove it from the list of basins2merge
            if parent_basin in basins2merge[:, 0]:
                if basins[parent_basin]['basin_area'] > thresh_area:
                    basins2merge = (
                        np.delete(basins2merge,
                                  np.where(basins2merge[:, 0]
                                           == parent_basin)[0],
                                  axis=0))
                # Else, update and sort the table again
                else:
                    basins2merge[
                        np.where(basins2merge[:, 0] == parent_basin)[0], 1]\
                        = basins[parent_basin]['basin_area']
                    basins2merge = basins2merge[basins2merge[:, 1].argsort()]

            basin_index += 1

        # print('Number of basins after filtering:', len(basins))
    # keep only the uper triangle of the adjacency matrix to avoid
    # the duplication of edges and ridges
    # k=1 excludes the diagonal to avoid possible self adjacency
    adjacency_triu = np.triu(adjacency, k=1)

    # Create a dictionary with ridge properties
    ridges = {}
    adjacency_index = np.array(np.where(adjacency_triu != 0)).T
    for i, j in adjacency_index:
        # ridge_vertices correspond to the indices of the vertices
        # located on the boundary between the two basins
        ridges_vertices = []
        for v in basins[i]['basin_vertices']:
            neighbors_vertices = np.array(vert_neigh[v])
            [ridges_vertices.append(v)
             for v in neighbors_vertices[vert_label[neighbors_vertices] == j]]
        ridges[(i, j)] = {}
        ridges[(i, j)]['ridge_vertices'] = np.unique(ridges_vertices)
        # ridge_length is the number of vertices in ridge_vertices
        # it is thus an approximation of the length of
        # the boundary between the two basins
        ridges[(i, j)]['ridge_length'] = len(ridges[(i, j)]['ridge_vertices'])
        # ridge_index is the index of the deepest point across ridge vertices
        # print((i, j))
        # print(ridges[(i, j)])
        # print(np.argmin(vert_depth[ridges[(i, j)]['ridge_vertices']]))
        ridges[(i, j)]['ridge_index'] = (
            ridges[(i, j)]['ridge_vertices'])[np.argmin(
                vert_depth[ridges[(i, j)]['ridge_vertices']])]
        # ridge_depth is the depth of the vertex at ridge_index
        ridges[(i, j)]['ridge_depth'] = np.min(
            vert_depth[ridges[(i, j)]['ridge_vertices']])
        # compute depth difference between ridge point and corresponding pits
        diff_i = abs(basins[i]['pit_depth'] - ridges[(i, j)]['ridge_depth'])
        diff_j = abs(basins[j]['pit_depth'] - ridges[(i, j)]['ridge_depth'])
        ridges[(i, j)]['ridge_depth_diff_min'] = min(diff_i, diff_j)
        ridges[(i, j)]['ridge_depth_diff_max'] = max(diff_i, diff_j)

    # for i, j in adjacency_index:
    #     print("-------------------------")
    #     print(i, " -> ", j)
    #     print(ridges[(i, j)]['ridge_vertices'])
    #     print('ridge_length =', ridges[(i, j)]['ridge_length'])
    #     print('ridge_index =', ridges[(i, j)]['ridge_index'])
    #     print('ridge_depth =', ridges[(i, j)]['ridge_depth'])
    #     print('ridge_depth_diff_min =',
    #           ridges[(i, j)]['ridge_depth_diff_min'])
    #     print('ridge_depth_diff_max =',
    #           ridges[(i, j)]['ridge_depth_diff_max'])
    #     print(j, " -> ", i)
    #     try:
    #         print(ridges[(j, i)]['ridge_vertices'])
    #         print('ridge_length =', ridges[(j, i)]['ridge_length'])
    #         print('ridge_index =', ridges[(j, i)]['ridge_index'])
    #         print('ridge_depth =', ridges[(j, i)]['ridge_depth'])
    #         print('ridge_depth_diff_min =',
    #               ridges[(j, i)]['ridge_depth_diff_min'])
    #         print('ridge_depth_diff_max =',
    #               ridges[(j, i)]['ridge_depth_diff_max'])
    #         depth_i = ridges[(i, j)]['ridge_depth']
    #         depth_j = ridges[(j, i)]['ridge_depth']
    #         print('depth_i<depth_j', depth_i < depth_j)
    #     except KeyError:
    #         print('edge does not exist, as expected')

    # reindex the adjacency matrix to delete removed basins
    basins_labels = list(basins.keys())
    adjacency_reind = adjacency_triu[basins_labels, :][:, basins_labels]
    return basins, ridges, adjacency_reind


def get_textures_from_dict(mesh, basins, ridges):
    """
        Function that returns the textures from the dictionaries outputs of
        the watershed
    Parameters
    ----------
    mesh
    basins
    ridges

    Returns
    -------
    atex_labels: numpy array
    atex_pits: numpy array
    atex_ridges: numpy array
    """
    vert = np.array(mesh.vertices)
    atex_labels = np.full(vert.shape[0], -1, dtype=np.int64)
    atex_pits = np.zeros(vert.shape[0], dtype=np.int64)
    atex_ridges = np.zeros(vert.shape[0], dtype=np.int64)

    for b in basins.keys():
        atex_labels[basins[b]['basin_vertices']] = b
        atex_pits[basins[b]['pit_index']] = 1

    for i, j in ridges:
        atex_ridges[ridges[(i, j)]['ridge_index']] = 1

    return atex_labels, atex_pits, atex_ridges


def get_texture_boundaries_from_dict(mesh, ridges):
    """
        Function that returns the textures from the dictionaries outputs of
    the watershed
    Parameters
    ----------
    mesh
    ridges

    Returns
    -------
    texture_boundaries
    """
    vert = np.array(mesh.vertices)
    # texture of basins boundaries
    atex_ridges_vert = np.zeros(vert.shape[0], dtype=np.int64)

    for i, j in ridges:
        atex_ridges_vert[ridges[(i, j)]['ridge_vertices']] = 1

    return atex_ridges_vert


def get_basins_attribute(basins, attribute='pit_index'):
    """
    get basins attribute as a list across basins.
    Parameters
    ----------
    basins
    attribute

    Returns
    -------

    """
    attr_out = list()
    for label, basin in basins.items():
        attr_out.append(basin[attribute])

    return attr_out


def get_ridges_attribute(ridges, attribute='ridge_index'):
    """
    get ridges attribute as a list across ridgess.
    Parameters
    ----------
    ridges
    attribute

    Returns
    -------

    """
    attr_out = list()
    for ind_tuple, ridge in ridges.items():
        attr_out.append(ridge[attribute])

    return attr_out
