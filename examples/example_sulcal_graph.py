###############################################################################
# importation of slam modules
import slam.io as sio
import slam.sulcal_graph as ssg
import numpy as np

###############################################################################
# loading an examplar mesh and corresponding texture
path_to_mesh = "../examples/data/example_mesh.gii"
path_to_texture = "../examples/data/example_texture.gii"
path_to_mask = ""
mesh = sio.load_mesh(path_to_mesh)
side = "left"
texture = sio.load_texture(path_to_texture)
dpf = np.array(texture.darray[0])

###############################################################################
# extract the sulcal graph from a mesh
g = ssg.extract_sulcal_graph(side, path_to_mesh, path_to_features=None, path_to_output=None, path_to_mask=None)

###############################################################################
# add an attribute to nodes
g = ssg.add_node_attribute_to_graph(g, dpf, name='pit_depth', save=False)
print("Node attributes:\n", g.nodes[0].keys())
print("First node:\n", g.nodes[0])

###############################################################################
# add an attribute to edges
g = ssg.add_edge_attribute_to_graph(g, dpf, name='ridge_depth_bis', save=False)
print("Edge attributes:\n", g.edges[list(g.edges)[0]].keys())
print("First edge:\n", g.edges[list(g.edges)[0]])

###############################################################################
# add geodesic distances attribute to edges
g = ssg.add_geodesic_distances_to_graph(g, mesh, save=False)

###############################################################################
# add mean value to nodes attributes
g = ssg.add_mean_value_to_graph(g, dpf, name='basin_mean_depth', save=False)

###############################################################################
# get textures from graph
tex_labels, tex_pits, tex_ridges = ssg.get_textures_from_graph(g, mesh, save=False, outdir=None)