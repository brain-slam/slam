import os
import numpy as np
from slam import io, differential_geometry, geodesics, texture, curvature, vertex_voronoi, sulcal_depth, utils


def compute_mesh_features(mesh, save=True, outdir=None, check_if_exist=True):
    """
    Function that computes the mean curvature, the depth potential function and the voronoi areas of a mesh.
    Returns numpy arrays.
    """

    if not outdir:
        outdir = ''

    path_to_mean_curvature = os.path.join(outdir, "mean_curvature.gii")
    path_to_dpf = os.path.join(outdir, "dpf.gii")
    path_to_voronoi = os.path.join(outdir, "voronoi.npy")

    # Get vertex mean curvature
    if check_if_exist and os.path.exists(path_to_mean_curvature):
        mean_curvature = io.load_texture(path_to_mean_curvature).darray[0]
    else:
        print("\n\tComputing the curvature\n")
        PrincipalCurvatures, _, _ = curvature.curvatures_and_derivatives(mesh)
        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        filt_mean_curv = (utils.z_score_filtering(np.array([mean_curv]), z_thresh=3))
        mean_curvature = filt_mean_curv[0]

    # Get vertex depth potential function
    if check_if_exist and os.path.exists(path_to_dpf):
        dpf = io.load_texture(path_to_dpf).darray
    else:
        print("\n\tComputing the DPF\n")
        # returns a list of dpf for different alpha values
        dpf = sulcal_depth.dpf_star(mesh, alphas=[0.03], curvature=mean_curvature)
        dpf = dpf[0]

    # Get vertex voronoi areas
    if check_if_exist and os.path.exists(path_to_voronoi):
        voronoi = np.load(path_to_voronoi)
    else:
        print("\n\tComputing Voronoi's vertex\n")
        voronoi = vertex_voronoi.vertex_voronoi(mesh)

    if save:
        curv_tex = texture.TextureND(darray=mean_curvature)
        io.write_texture(curv_tex, path_to_mean_curvature)
        dpf_tex = texture.TextureND(darray=dpf)
        io.write_texture(dpf_tex, path_to_dpf)
        np.save(path_to_voronoi, voronoi)

    return mean_curvature, dpf, voronoi


def normalize_thresholds(mesh, voronoi, thresh_dist=20.0, thresh_ridge=1.5, thresh_area=50.0, side="left"):
    """
    Function that normalizes the thresholds for the watershed algorithm.
    Threshold on distance between pits is normalized by the Fiedler geodesic length.
    Threshold on basin's area is normalized by the surface area of the mesh.

    :args: mesh: surface triangular mesh loaded with io.load_mesh() (trimesh object)
    :args: voronoi: voronoi areas of the mesh (numpy array)
    :args: thresh_dist: distance threshold for the watershed algorithm (float)
    :args: thresh_ridge: ridge threshold for the watershed algorithm (float)
    :args: thresh_area: area threshold for the watershed algorithm (float)
    :args: side: side of the brain (left or right)
    :return: normalized thresholds (thresh_dist, thresh_ridge, thresh_area)

    """

    # Compute Fiedler length and surface area for watershed threshold normalization
    print("\n\tComputing the Fiedler geodesic length and surface area\n")
    fielder = differential_geometry.mesh_laplacian_eigenvectors(mesh, 1)
    imin = fielder.argmin()
    imax = fielder.argmax()
    min_mesh_fiedler_length = geodesics.compute_gdist(mesh, imin, imax)
    mesh_area = np.sum(voronoi)

    # Set group average values
    group_average_fiedler_length = 238.25 if side == "right" else 235.95
    group_average_surface_area = 91433.68 if side == "right" else 91369.33

    # Normalize the thresholds regarding brain size
    thresh_dist *= min_mesh_fiedler_length / group_average_fiedler_length
    thresh_area *= mesh_area / group_average_surface_area

    print("Fielder length", min_mesh_fiedler_length)
    print("Distance threshold :", thresh_dist)

    return thresh_dist, thresh_ridge, thresh_area


def watershed(mesh, voronoi, dpf, thresh_dist, thresh_ridge, thresh_area, mask=None):
    """
    Sulcal segmentation and sulcal pits extraction using watershed by flooding.

    Reference paper is:
    G. Auzias, L.Brun, C. Deruelle, O. Coulon. 2015. Deep sulcal landmarks: algorithmic and conceptual improvements in the definition and extraction of sulcal pits. NeuroImage

    A related approach can be found in:
    Rettmann ME, Han X, Xu C, Prince JL. 2002. Automated sulcal segmentation using watersheds on the cortical surface. Neuroimage. 15:329-344.

    INPUTS

    mesh : white matter triangular mesh of subject (trimesh object)
    voronoi : voronoi area for each vertex (numpy array)
    dpf : depth measure for each vertex (numpy array)
    mask : binary array for cingular pole exclusion (with ones in the region to exclude)
    thresh_dist : threshold on the distance between pits (unit: mm)
    thresh_ridge : threshold on the ridge height (unit: mm)
    thresh_area : threshold on the basin area (unit: mm²)

    OUTPUTS

    basins : dictionary with basin properties
        basins[label] = {}
            'pit_index': vertex index of the sulcal pit
            'pit_depth': depth of the sulcal pit
            'basin_vertices': list of vertices in the basin
            'basin_area': area of the basin
    ridges : dictionary with properties of the ridge between basin i and basin j
        ridges[(i,j)] = {}
            'ridge_index': vertex index of the ridge point
            'ridge_depth': depth of the ridge point
            'ridge_length': number of vertices along the ridge
    adjacency : adjacency matrix of the basins
        adjacency[i,j] = 1 if basin i and j are adjacent, 0 otherwise

    """

    print('Computing watershed by flooding...')
    print(f'Distance between 2 pits: {thresh_dist} mm - Ridge height: {thresh_ridge} mm')

    # Initialize output tables and dictionaries
    n_vertices = mesh.vertices.shape[0]  # mesh.vertices returns vertices coordinates (n, 3) float
    vert_idx = np.arange(n_vertices)  # vertex index
    vert_depth = dpf.reshape(n_vertices)
    vert_area = voronoi.reshape(n_vertices)
    vert_label = np.full(n_vertices, -1, dtype=np.int64)  # vertex labels (initialized to -1)
    vert_neigh = mesh.vertex_neighbors  # list of lists of neighbors
    basins = {}
    adjacency = np.zeros((1, 1), dtype=np.int64)

    # Apply exclusion mask
    # All nodes included in the exclusion mask are not taken into account in the watershed process
    if not mask:
        mask = np.zeros(dpf.shape)
    mask_indices = np.where(mask == 1)[0]
    nodes = np.vstack((vert_idx, vert_depth)).T
    nodes = np.delete(nodes, mask_indices, axis=0)

    # Sorting step
    # All nodes of the mesh are sorted by their depth (deepest nodes = highest values first)
    nodes_sorted_by_depth = nodes[nodes[:, 1].argsort()[::-1]]

    # Initialize with first pit: deepest node
    idx = int(nodes_sorted_by_depth[0, 0])
    new_label = 0
    vert_label[idx] = new_label
    basins[0] = {}
    basins[0]['pit_index'] = idx
    basins[0]['basin_vertices'] = [idx]

    for node in nodes_sorted_by_depth[1:]:

        idx = int(node[0])
        neigh_idx = np.array(vert_neigh[idx]).astype(int)  # indices of neighbors
        neigh_labels = vert_label[neigh_idx]  # labels of neighbors
        # Keep only labelled neighbors
        neigh_idx = neigh_idx[:, None][neigh_labels != -1]
        neigh_labels = neigh_labels[neigh_labels != -1]
        NL = np.unique(neigh_labels)

        #####################################################################################################
        # Case 1: all of its neighbors are unlabeled. Then, this node corresponds to the deepest point of a new
        # catchment basin.
        #####################################################################################################
        if len(NL) == 0:
            new_label += 1  # max(basins.keys()) + 1
            # Update vertex
            vert_label[idx] = new_label
            # Update basins
            basins[new_label] = {}
            basins[new_label]['pit_index'] = idx
            basins[new_label]['basin_vertices'] = [idx]
            # Update adjacency matrix: increment size by 1
            adjacency = np.pad(adjacency, ((0, 1), (0, 1)), mode='constant', constant_values=0)

        #####################################################################################################
        # Case 2: the node is the neighbor of only one catchment basin. Then, this node is assigned to the
        # corresponding basin.
        #####################################################################################################
        elif len(NL) == 1:
            # Update vertex
            vert_label[idx] = NL[0]
            # Update basin
            basins[NL[0]]['basin_vertices'].append(idx)

        #####################################################################################################
        # Case 3: the node is the neighbor of two or more catchment basins. Then, this node is a ridge point where
        # each pair of basins join.
        # It is assigned to the basin represented by the deepest neighbor vertex, or the lowest label if same depth.
        # Then the conditions for merging two basins are tested (distance between the two pits and ridge height).
        #####################################################################################################
        else:
            idx_max_depth = np.argmax(vert_depth[neigh_idx])  # deepest neighbor
            lab = np.min(neigh_labels[idx_max_depth])  # lowest label
            # Update vertex
            vert_label[idx] = lab
            # Update basin
            basins[lab]['basin_vertices'].append(idx)

            # MERGING between pairs of neighbor catchment basins
            # The merging condition is questioned between the intersection label (lab) and all other neighbors
            # The merging condition is only questioned if the basins have never met yet (no existing ridge point)

            # NL.sort()  # NL is already sorted when using np.unique()
            NL = np.delete(NL, NL == lab)
            for neighbor_lab in NL:
                # Check that basins have not already been merged in the loop
                if neighbor_lab in np.unique(vert_label) and lab in np.unique(vert_label):
                    # Forcing label_i < label_j so that pit(i) deeper than pit(j)
                    label_i, label_j = min(lab, neighbor_lab), max(lab, neighbor_lab)
                    if adjacency[label_i, label_j] == 0:  # first meet

                        merging = 0

                        # Compute ridge height
                        ridge_height = abs(vert_depth[basins[label_j]['pit_index']] - node[1])
                        if ridge_height < thresh_ridge:

                            # Compute distance between pits
                            v = geodesics.compute_gdist(mesh,
                                                        basins[label_i]['pit_index'],
                                                        basins[label_j]['pit_index'])
                            if v < thresh_dist:
                                # print('merging of', label_j, 'into', label_i)
                                merging = 1
                                # Update vertex
                                vert_label[np.where(vert_label == label_j)[0]] = label_i
                                # Update basin
                                [basins[label_i]['basin_vertices'].append(b) for b in basins[label_j]['basin_vertices']]
                                del basins[label_j]
                                # Update ridges
                                # Add adjacent basins of j to i and clean out adjacency of j
                                adjacency[label_i, :] = adjacency[label_i, :] | adjacency[label_j, :]
                                adjacency[:, label_i] = adjacency[:, label_i] | adjacency[:, label_j]
                                adjacency[label_j, :] = 0
                                adjacency[:, label_j] = 0

                                if label_j == lab:
                                    # lab disappears and is replaced by the parent label which becomes adjacent to other
                                    # neighbors. Continue the loop to test merge with this new lab.
                                    lab = neighbor_lab

                        # Create a new entry in the ridge dictionary
                        if not merging:
                            adjacency[label_i, label_j] = adjacency[label_j, label_i] = 1

    print('Number of basins found:', len(basins))

    #####################################################################################################
    # FILTERING OF SMALL BASINS
    # Last merging step on area criterion after the watershed process
    # Each basin which area is less than thresh_area is merged into the neighbor basin it shares the
    # largest border with.
    #####################################################################################################

    # Select basins with area < thresh_area
    basins2merge = []
    for label in basins:
        basins[label]['pit_depth'] = vert_depth[basins[label]['pit_index']]
        basins[label]['basin_area'] = np.sum(vert_area[basins[label]['basin_vertices']])
        if basins[label]['basin_area'] <= thresh_area:
            basins2merge.append([label, basins[label]['basin_area']])
    basins2merge = np.array(basins2merge)

    print('nb of basins to remove:', basins2merge.shape[0])
    if basins2merge.shape[0] > 0:

        print('Filtering of small basins...')

        # Sort by area
        basins2merge = basins2merge[basins2merge[:, 1].argsort()]

        # Filtering
        basin_index = 0
        while basin_index != len(basins2merge):
            basin, area = basins2merge[basin_index]
            basin = int(basin)

            # Neighbor basins are those sharing a ridge point with current basin
            neigh_basins = np.where(adjacency[basin] != 0)[0]
            ridges_length = []
            for nb in neigh_basins:
                # List all ridge points
                ridges_vertices = []
                for v in basins[basin]['basin_vertices']:
                    neighbors_vertices = np.array(vert_neigh[v])
                    [ridges_vertices.append(v) for v in neighbors_vertices[vert_label[neighbors_vertices] == nb]]
                # Get ridge length with each neighbor
                ridges_length.append(len(ridges_vertices))
            # Parent basin: the one sharing the largest border
            parent_basin = np.min(neigh_basins[np.argmax(ridges_length)])

            # print('merging of', basin, 'into', int(parent_basin))

            # Update vertex
            vert_label[np.where(vert_label == basin)[0]] = parent_basin
            # Update basin
            [basins[parent_basin]['basin_vertices'].append(b) for b in basins[basin]['basin_vertices']]
            basins[parent_basin]['basin_area'] += area
            del basins[basin]
            # Update ridges matrix
            adjacency[parent_basin, :] = adjacency[parent_basin, :] | adjacency[basin, :]
            adjacency[:, parent_basin] = adjacency[:, parent_basin] | adjacency[:, basin]
            adjacency[basin, :] = 0
            adjacency[:, basin] = 0

            # Update basins2merge
            if parent_basin in basins2merge[:, 0]:
                # If new area of parent basin is now above the threshold, remove it from the list of basin to merge
                if basins[parent_basin]['basin_area'] > thresh_area:
                    basins2merge = np.delete(basins2merge, np.where(basins2merge[:, 0] == parent_basin)[0], axis=0)
                # Else, update its area and sort the table again
                else:
                    basins2merge[np.where(basins2merge[:, 0] == parent_basin)[0], 1] = basins[parent_basin][
                        'basin_area']
                    basins2merge = basins2merge[basins2merge[:, 1].argsort()]

            basin_index += 1

        print('Number of basins after filtering:', len(basins))

    # Create a dictionary with ridge properties
    ridges = {}
    adjacency_index = np.array(np.where(adjacency != 0)).T
    for i, j in adjacency_index:
        # list all ridge points between basin i and basin j
        ridges_vertices = []
        for v in basins[i]['basin_vertices']:
            neighbors_vertices = np.array(vert_neigh[v])
            [ridges_vertices.append(v) for v in neighbors_vertices[vert_label[neighbors_vertices] == j]]
        ridges[(i, j)] = {}
        ridges[(i, j)]['ridge_index'] = ridges_vertices[np.argmax(vert_depth[ridges_vertices])]
        ridges[(i, j)]['ridge_depth'] = np.max(vert_depth[ridges_vertices])
        ridges[(i, j)]['ridge_length'] = len(ridges_vertices)

    return basins, ridges, adjacency


def get_textures_from_dict(mesh, basins, ridges, save=True, outdir=None):
    """
    Function that returns the textures from the dictionaries outputs of the watershed
    """
    if save and not outdir:
        outdir = ''

    vert = np.array(mesh.vertices)
    labels = np.full(vert.shape[0], -1, dtype=np.int64)
    atex_pits = np.zeros(vert.shape[0])
    atex_ridges = np.zeros(vert.shape[0])

    for b in basins.keys():
        labels[basins[b]['basin_vertices']] = b
        atex_pits[basins[b]['pit_index']] = 1

    for i, j in ridges:
        atex_ridges[ridges[(i, j)]['ridge_index']] = 1

    # texture of labels
    tex_labels = texture.TextureND(darray=labels.flatten())
    if save:
        io.write_texture(tex_labels, os.path.join(outdir, "labels.gii"))

    # texture of pits
    tex_pits = texture.TextureND(darray=atex_pits.flatten())
    if save:
        io.write_texture(tex_pits, os.path.join(outdir, "pits_tex.gii"))

    # texture of ridges
    tex_ridges = texture.TextureND(darray=atex_ridges.flatten())
    if save:
        io.write_texture(tex_ridges, os.path.join(outdir, "rigdes_tex.gii"))

    return tex_labels, tex_pits, tex_ridges
