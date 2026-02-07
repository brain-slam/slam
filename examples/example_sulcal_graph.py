"""
.. _example_sulcal_graph:

===================================
Example of sulcal graph in slam
===================================
"""

# Authors:
# Lucile Hashimoto lucile-hashimoto
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# importation of slam modules
import numpy as np
import slam.io as sio
import slam.sulcal_graph as ssg
import slam.watershed as swat


###############################################################################
# loading an examplar mesh and corresponding texture
path_to_mesh = "../examples/data/example_mesh.gii"
path_to_texture = "../examples/data/example_texture.gii"
path_to_mask = None
mesh = sio.load_mesh(path_to_mesh)
side = "left"
texture = sio.load_texture(path_to_texture)
dpf = np.array(texture.darray[0])

# define the exclusion mask (cingular pole)
if path_to_mask is not None:
    mask = sio.load_texture(path_to_mask).darray[0]
else:
    mask = None

###############################################################################
# extract the sulcal graph from a mesh
g = ssg.extract_sulcal_graph(side, mesh, mask=mask)

###############################################################################
# add an attribute to nodes
g = ssg.add_node_attribute_to_graph(g, dpf, attribute_name='pit_depth')
print("Node attributes:\n", g.nodes[0].keys())
print("First node:\n", g.nodes[0])

###############################################################################
# add an attribute to edges
g = ssg.add_edge_attribute_to_graph(g, dpf, attribute_name='ridge_depth_bis')
print("Edge attributes:\n", g.edges[list(g.edges)[0]].keys())
print("First edge:\n", g.edges[list(g.edges)[0]])

###############################################################################
# add geodesic distances attribute to edges
g = ssg.add_geodesic_distances_to_graph(g, mesh)

###############################################################################
# add mean value to nodes attributes
g = ssg.add_mean_value_to_graph(g, dpf, attribute_name='basin_mean_depth')

###############################################################################
# get textures from graph
atex_labels, atex_pits, atex_ridges = ssg.get_textures_from_graph(g, mesh)

###############################################################################
# A more detailed computation of the sulcal graph
# with explicit call to the watershed
###############################################################################
# compute curvature, dpf and voronoi
mean_curvature, dpf, voronoi = swat.compute_mesh_features(mesh)

###############################################################################
# normalize watershed thresholds
thresh_dist, thresh_ridge, thresh_area = swat.normalize_thresholds(voronoi, thresh_dist=20.0, thresh_ridge=1.5,
                                                                   thresh_area=50.0, side=side)
# extract sulcal pits and associated basins
basins, ridges, adjacency = swat.watershed(
    mesh, voronoi, dpf, thresh_dist, thresh_ridge, thresh_area, mask)

# generate the sulcal graph
g = ssg.get_sulcal_graph(adjacency, basins, ridges)

###############################################################################
# generate the textures from graph
atex_labels_graph, atex_pits_graph, atex_ridges_graph = (ssg.get_textures_from_graph
                                    (g, mesh))
# compare the textures extracted from the watershed with the ones extarcted from the graph
# they should be identical
print("vertex-to-vertex difference between the texture extracted from the "
      "watershed and the one extracted from the graph, should be strictly qual to 0")
print(np.max(atex_ridges-atex_ridges_graph))

