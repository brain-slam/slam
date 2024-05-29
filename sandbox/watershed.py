import numpy as np
import slam.geodesics as geodesics

def watershed(white, vert_area, depthArray, mask, threshDist, threshRidge):
    # Sulcal segmentation and sulcal pits extraction using watershed by flooding.
    # reference paper is
    # G. Auzias, L.Brun, C. Deruelle, O. Coulon. 2015. Deep sulcal landmarks: algorithmic and conceptual improvements in the definition and extraction of sulcal pits. NeuroImage
    #
    # a related approach can be found in:
    # Rettmann ME, Han X, Xu C, Prince JL. 2002. Automated sulcal segmentation using watersheds on the cortical surface. Neuroimage. 15:329-344.

    # INPUTS
    #
    # mesh : white matter triangular mesh of subject (gifti)
    # voronoi : texture where each label corresponds to the vertex area
    # depthMap : texture of depth measure (tex)
    # maskTexture : binary texture for cingular pole (tex)
    # threshDist, threshRidge : thresholds for the merging of sulcal basins (unit: mm)
    #
    # OUTPUTS
    #
    # labels : list of vertices labels
    # pitsKept : list of [index, depth] of remaining pits after watershed
    # pitsRemoved : list of [index, depth] of pits merged in the filtering process
    # ridgePoints : list of ridge points and both related basins [label1, label2, ridge point index, ridge point depth]
    # parent : parent[i] is basin's label in which basin i has been merged, else parent[i]=i

    # print 'Computing watershed by flooding'

    # print 'Distance between 2 pits:', threshDist, 'mm - Ridge height:', threshRidge, 'mm'

    # vert = np.array(white.vertices())  # vertices coordinates
    vert = np.array(white.vertices)
    neigh = white.vertex_neighbors

    # neigh = aims.SurfaceManip.surfaceNeighbours(white)  # vertices neighborhood
    # g = aims.GeodesicPath(white, 3, 0)  # compute shortest path between each pair of vertex

    idx = np.arange(vert.shape[0]).reshape(vert.shape[0], 1)  # vertices index
    depthArray = depthArray.reshape(vert.shape[0], 1)
    nodes = np.concatenate((idx, depthArray), axis=1)

    ## Apply exclusion mask
    # All nodes included in the exclusion mask are not taken into acount in the watershed process
    maskIndices = np.where(mask == 1)[0]
    nodes = np.delete(nodes, maskIndices, axis=0)

    ## Sorting stepwœ
    # All nodes of the mesh are sorted by their depth (deepest nodes = highest values first)
    sorted_nodes = np.array(sorted(nodes, key=lambda depth: depth[1], reverse=True))

    ## Flooding & Merging step
    pitsAll = []  # List of pits. Will contain [pit's index, pit's depth]
    pitsRemoved = []
    labels = np.zeros((vert.shape[0], 1),
                      dtype=np.int64) - 1  # List of vertex labels (set to -1). Will be set to pit's index of basin
    labels_unmerged = np.zeros((vert.shape[0], 1), dtype=np.int64) - 1
    labels_merged = []
    parent = np.zeros((1, 1), dtype=np.int64) - 1  # List of each pit's direct parent (set to -1)
    ridges = np.zeros((1, 1), dtype=np.int64)  # List of ridge points

    # First pit: deepest node
    pitsAll.append(sorted_nodes[0])
    labels[int(sorted_nodes[0, 0])] = 0
    labels_unmerged[int(sorted_nodes[0, 0])] = 0

    for node in sorted_nodes[1:]:

        nind = int(node[0])
        neighbors = neigh[nind]  # indices of neighbors
        neigh_labels = []
        neigh_nodes = []

        for n in neighbors:
            if (labels[n] != -1):
                # neighbor's that have already been labelled
                neigh_labels.append(labels[n][0])
                neigh_nodes.append(n)
        NL = list(set(neigh_labels))  # labels occurences

        ## Case 1: all of its neighbors are unlabeled. Then, this node corresponds to the deepest point of a new catchment basin.
        if len(NL) == 0:
            # print 'new pit:', len(pitsAll)
            labels[nind] = len(pitsAll)
            labels_unmerged[nind] = len(pitsAll)
            pitsAll.append(node)
            # Allocation of space for a new potential ridge point (and parent)
            ridges = np.concatenate((ridges,
                                     np.zeros((1, len(ridges)),
                                              dtype=np.int64)), axis=0)
            ridges = np.concatenate((ridges, np.
                                     zeros((len(ridges), 1),
                                           dtype=np.int64)), axis=1)
            parent = np.concatenate((parent, np.zeros((1, 1), dtype=np.int64) - 1), axis=0)

        ## Case 2: the node is the neighbor of only one catchment basin. Then, this node is assigned to the corresponding basin.
        elif len(NL) == 1:
            lab = NL
            labels[nind] = lab
            labels_unmerged[nind] = lab


        ## Case 3: the node is the neighbor of two or more catchment basins. Then, this node is a ridge point where each pair of basins join.
        # It is assigned to the basin represented by the deepest neighbor vertex, or the lowest label if same depth.
        # Then the conditions for merging two basins are tested (distance between the two pits and ridge height).
        else:
            indx_max = np.argmax(depthArray[neigh_nodes])
            lab = np.min(np.array(neigh_labels)[indx_max])  # lowest label
            labels[nind] = lab

            # MERGING between pairs of neighbor catchment basins
            # NB: by construction, labels are ordered by pits depth (if i<j, pit(i) deeper than pit(j)).
            # So if the merging condition is met, then basin j is merged into basin i
            # The merging condition is only questioned if the basins have never met yet (no existing ridge point)
            NL.sort()
            X = 1  # starting index in NL for second loop (label_j)
            for label_i in NL[:len(NL) - 1]:
                # skip basins that have already been merged in the loop
                if label_i in labels_merged:
                    X += 1
                else:
                    for label_j in NL[X:]:
                        if ridges[label_i][label_j] == 0:
                            # create the ridge point
                            ridges[label_i][label_j] = ridges[label_j][label_i] = nind
                            # compute ridge height
                            ridge_height = abs(pitsAll[label_j][1] - node[1])

                            if ridge_height < threshRidge:
                                # compute distance between pits
                                # v = aims.vector_FLOAT()  # vector of distances from shallower pit to merge
                                # g.distanceMap_1_N_ind(int(pitsAll[label_i][0]), v, 0)

                                v = geodesics.compute_gdist(white, int(pitsAll[label_i][0]))

                                if v[int(pitsAll[label_j][0])] < threshDist:
                                    print('merging of', label_j, 'into', label_i)
                                    labels_merged.append(label_j)
                                    pitsRemoved.append(pitsAll[label_j])
                                    labels[np.where(labels == label_j)[0]] = label_i  # update all vertices labels
                                    parent[label_j] = label_i  # assign direct parent label

                                    # Exception: new node has been assigned to a label different from the two that
                                    # are merging. This leads to a wrong frontier between the two former basins.
                                    # Solution: re-assign the node to the merged basin.
                                    if label_i != lab and label_j != lab:
                                        labels[nind] = label_i
                    X += 1

    ## Results

    # parents of unmerged pits : parent[i]=i
    for i in range(parent.size):
        if (parent[i] == -1):
            parent[i] = i

    nbPits = len(pitsAll)
    # print 'Number of pits found:', nbPits

    # Create a new list "pitsKept" with all remaining pits of unmerged basins
    indx = []
    pitsKept = np.copy(pitsAll)
    for pit in pitsRemoved:
        indx.append(np.where(pitsKept[:, 0] == pit[0])[0])
    tmp = np.delete(pitsKept, indx, axis=0)
    pitsKept = tmp

    nbFinalPits = len(pitsKept)
    # print 'Number of pits kept:', nbFinalPits

    # List of ridge points and both related basins [label1, label2, ridge point index, ridge point depth]
    basins = np.unique(labels)
    basins = np.delete(basins, np.where(basins == -1)[0])
    ridgePoints = []
    for label_i in basins:
        for label_j in basins:
            if (label_j > label_i and ridges[label_i][label_j] != 0):
                ridgePoints.append(
                    [label_i, label_j, ridges[label_i][label_j], float(depthArray[ridges[label_i][label_j]])])

    # sys.stdout=orig_stdout
    # f.close()
    return labels, pitsKept, pitsRemoved, ridgePoints, parent


def areaFiltering(mesh, vert_area, labels, pitsKept, parent, threshArea):
    # Last merging step on area criterion after the watershed process
    # Each basin which area is less than threshArea is merged into the neighbor basin it shares the largest border with
    #
    # INPUTS
    #
    # mesh : white matter triangular mesh of subject (gifti)
    # voronoi : texture where each label corresponds to the vertex area
    # pitsKept : temporary list of [index, depth] of remaining pits after the watershed
    # parent : temporary list of parents for each basin after watershed
    # threshArea : threshold for the merging of small sulcal basins (unit: mm2)
    #
    # OUTPUTS
    #
    # labels : list of vertices labels
    # infoBasins : array of final basins informations [label, pit's index, pit's depth, basin area]
    # pitsKept : list of [index, depth] of final pits
    # pitsRemoved : list of [index, depth] of pits merged in the filtering process (should be added to the previous output of "watershed")
    # parent : parent[i] is basin's label in which basin i has been merged, else parent[i]=i

    # threshArea=float(threshArea)
    # print 'Area threshold:', threshArea

    # neigh = aims.SurfaceManip.surfaceNeighbours(mesh)
    neigh = mesh.vertex_neighbors

    tmp = np.unique(labels)
    tmp = np.delete(tmp, np.where(tmp == -1)[0])
    tmp = np.reshape(tmp, (len(tmp), 1))
    basinsArray = np.concatenate((tmp, np.zeros((len(tmp), 2), dtype=int)), axis=1)  # will contain [label, area, pit]
    # Compute basins area
    # print 'basins area'
    for b in basinsArray:
        b[1] = np.sum(vert_area[np.where(labels == b[0])[0]])
        b[2] = pitsKept[np.where(labels[pitsKept[:, 0].astype(int).tolist()] == b[0])[0][0]][0]
    # sort by area
    sorted_basins = np.array(sorted(basinsArray, key=lambda area: area[1], reverse=False))
    basins2merge = sorted_basins[np.where(sorted_basins[:, 1] <= threshArea)[0]]
    # print 'nb of basins to remove:', len(basins2merge)
    # Filtering
    # print 'filtering...'
    pitsRemoved = []
    basin_index = 0
    while basin_index != len(basins2merge):
        basin = basins2merge[basin_index]
        # list of its neighbors
        neigh_labels = []
        neigh_nodes = []
        verts = np.where(labels == basin[0])[0]
        for v in verts:
            neighbors = neigh[v]
            for n in neighbors:
                if (labels[n] != -1 and labels[n] != basin[0]):
                    neigh_labels.append(float(labels[n]))
                    neigh_nodes.append(n)
        neigh_nodes = np.array(neigh_nodes)
        # set of neighbor labels
        NL = np.unique(neigh_labels)
        # parent basin: the one sharing the largest border
        borders_length = []
        for nl in NL:
            border_nl = set(neigh_nodes[np.where(neigh_labels == nl)[0]])
            borders_length.append(len(border_nl))
        indx = np.argmax(borders_length)
        parent_basin = NL[np.min(indx)]
        # print 'merging of', int(basin[0]), 'into', int(parent_basin)
        pitsRemoved.append(pitsKept[np.where(pitsKept[:, 0].astype(int).tolist() == basin[2])[0]][0])
        labels[np.where(labels == basin[0])[0]] = parent_basin
        parent[basin[0]] = parent_basin
        # update basins2merge
        basins2merge[basin_index, 1] = 0
        basin2update = np.where(basins2merge[:, 0] == parent_basin)[0]
        if len(basin2update) != 0:
            new_area = np.sum(vert_area[np.where(labels == parent_basin)[0]])
            if new_area > threshArea:
                basins2merge = np.delete(basins2merge, basin2update, axis=0)
                # print 'basin', int(parent_basin), 'area is now above the threshold'
            else:
                basins2merge[basin2update, 1] = new_area
                basins2merge = np.array(sorted(basins2merge, key=lambda area: area[1], reverse=False))
        basin_index += 1

    indx = []
    for pit in pitsRemoved:
        indx.append(np.where(pitsKept[:, 0].astype(int) == pit[0])[0])
    tmp = np.delete(pitsKept, indx, axis=0)
    pitsKept = tmp

    nbFinalPits = len(pitsKept)
    # print 'Number of pits kept after filtering:', nbFinalPits

    # Array of final basins info [label, pit's index, pit's depth, basin area]
    basins = labels[pitsKept[:, 0].astype(int).tolist()].reshape((1, len(labels[pitsKept[:, 0].astype(int).tolist()])))[
        0]
    infoBasins = np.zeros((len(pitsKept), 4))
    for i in range(len(pitsKept)):
        infoBasins[i][0] = int(basins[i])
        infoBasins[i][1] = int(pitsKept[i][0])
        infoBasins[i][2] = pitsKept[i][1]
        infoBasins[i][3] = np.sum(vert_area[np.where(labels == basins[i])[0]])

    return labels, infoBasins, pitsKept, pitsRemoved, parent