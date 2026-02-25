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
import networkx as nx
import slam.io as sio
import slam.sulcal_graph as ssg
import slam.watershed as swat
import slam.remeshing as srem


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
g = ssg.add_node_attribute_from_texture(g, dpf, attribute_name='pit_depth')
print("Node attributes:\n", g.nodes[0].keys())
print("First node:\n", g.nodes[0])

###############################################################################
# add an attribute to edges
g = ssg.add_edge_attribute_from_texture(g, dpf, attribute_name='ridge_depth_bis')
print("Edge attributes:\n", g.edges[list(g.edges)[0]].keys())
print("First edge:\n", g.edges[list(g.edges)[0]])

###############################################################################
# add geodesic distances attribute to edges
g = ssg.add_geodesic_distances_to_edges(g, mesh)

###############################################################################
# add mean value to nodes attributes
g = ssg.add_mean_value_to_nodes(g, dpf, attribute_name='basin_mean_depth')

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

###############################################################################
# add as a new node attribute the 3D coordinates of the vertex corresponding to
# the pit in the mesh used to compute the watershed
g = ssg.add_coords_attribute(g, mesh,
                                    attribute_vert_index='pit_index',
                                    new_attribute_key='3dcoords')
print("First node:\n", g.nodes[0])

###############################################################################
# add as a new node attribute the 3D coordinates of the vertex corresponding to
# the pit in the spherical mesh obtained from the original mesh, so that 'pit_index'
# also gives the corresponding vertex in that mesh
source_spherical_mesh_file = "../examples/data/example_mesh_spherical.gii"
source_spherical_mesh = sio.load_mesh(source_spherical_mesh_file)

g = ssg.add_coords_attribute(g, source_spherical_mesh,
                                    attribute_vert_index='pit_index',
                                    new_attribute_key='sphere_3dcoords')
print("First node:\n", g.nodes[0])

###############################################################################
# Load another mesh and corresponding spherical version to be used as a target
# onto which the graph will be projected
target_mesh_file = "../examples/data/example_mesh_2.gii"
target_mesh = sio.load_mesh(target_mesh_file)
target_spherical_mesh_file = "../examples/data/example_mesh_2_spherical.gii"
target_spherical_mesh = sio.load_mesh(target_spherical_mesh_file)

###############################################################################
# Project the depth texture using
# resampling.texture_spherical_interpolation_nearest_neighbor for the visu
interpolated_dpf = srem.texture_spherical_interpolation_nearest_neighbor(
    source_spherical_mesh, target_spherical_mesh, dpf)

###############################################################################
# Compute the 'interpolated_pits_index' corresponding to the index of the nearest neighbor
# of each pit in the target spherical mesh
g = ssg.vertex_index_interpolation(g, target_spherical_mesh,
                                              graph_spherical_coords_attribute='sphere_3dcoords',
                                              interpolated_attribute='interpolated_pits_index')
print("First node:\n", g.nodes[0])

###############################################################################
# Here is the way to get the list of 'interpolated_pits_index' for all nodes
interp_pits_inds = np.array(list(nx.get_node_attributes(g, 'interpolated_pits_index').values()))

###############################################################################
# add as a new node attribute the 3D coordinates of the vertex corresponding to
# # the inperpolated pit in the target spherical mesh
g = ssg.add_coords_attribute(g, target_spherical_mesh,
                                    attribute_vert_index='interpolated_pits_index',
                                    new_attribute_key='target_sphere_3dcoords')
print("First node:\n", g.nodes[0])

###############################################################################
# add as a new node attribute the 3D coordinates of the vertex corresponding to
# # the inperpolated pit in the target mesh
g = ssg.add_coords_attribute(g, target_mesh,
                                    attribute_vert_index='interpolated_pits_index',
                                    new_attribute_key='target_mesh_3dcoords')
print("First node:\n", g.nodes[0])

#############################################################################
# VISUALIZATION USING plotly
#############################################################################

import slam.plot as splt

mesh_data = {
    "vertices": mesh.vertices,
    "faces": mesh.faces,
    "title": 'Source'
}
intensity_data = {
    "values": dpf,
    "mode": "vertex",
}
fig1 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data)
# add the pits to the plot
pits_coords = np.array(list(nx.get_node_attributes(g, '3dcoords').values()))
trace_hover = splt.create_hover_trace(
    pits_coords,
    marker={"size": 6, "color": "black"},
)
fig1.add_trace(trace_hover)
fig1.show()
fig1


mesh_data = {
    "vertices": source_spherical_mesh.vertices,
    "faces": source_spherical_mesh.faces,
    "title": 'Source spherical'
}
fig2 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data)
# add the pits to the plot
pits_coords = np.array(list(nx.get_node_attributes(g, 'sphere_3dcoords').values()))
trace_hover = splt.create_hover_trace(
    pits_coords,
    marker={"size": 6, "color": "black"},
)
fig2.add_trace(trace_hover)
fig2.show()
fig2


mesh_data = {
    "vertices": target_spherical_mesh.vertices,
    "faces": target_spherical_mesh.faces,
    "title": 'Target sphere'
}
intensity_data = {
    "values": interpolated_dpf,
    "mode": "vertex",
}
fig3 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data)
# add the pits to the plot
pits_init = splt.create_hover_trace(
    pits_coords,
    marker={"size": 6, "color": "black"},
)
fig3.add_trace(pits_init)
# add the interpolated pits to the plot
interp_pits_coords_sphere = np.array(list(nx.get_node_attributes(g, 'target_sphere_3dcoords').values()))
pits_interpolated = splt.create_hover_trace(
    interp_pits_coords_sphere,
    marker={"size": 6, "color": "red"},
)
fig3.add_trace(pits_interpolated)
fig3.show()
fig3


mesh_data = {
    "vertices": target_mesh.vertices,
    "faces": target_mesh.faces,
    "title": 'Target mesh'
}
intensity_data = {
    "values": interpolated_dpf,
    "mode": "vertex",
}
fig4 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data)
# add the interpolated pits to the plot
interp_pits_coords_mesh = np.array(list(nx.get_node_attributes(g, 'target_mesh_3dcoords').values()))
pits_interpolated = splt.create_hover_trace(
    interp_pits_coords_mesh,
    marker={"size": 6, "color": "red"},
)
fig4.add_trace(pits_interpolated)
fig4.show()
fig4



