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

thresh_dist_perc=0
thresh_ridge=0
thresh_area_perc=0
###############################################################################
# extract the sulcal graph from a mesh
g = ssg.extract_sulcal_graph(mesh,
                             thresh_dist_perc=thresh_dist_perc,
                             thresh_ridge=thresh_ridge,
                             thresh_area_perc=thresh_area_perc,
                             mask=mask)

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
##### compute Euclidian distance between pits
print("compute Euclidian dist")
g = ssg.add_edge_attribute_euclidian_distances_to_ridge(g,mesh)

##### compute geodesic distance between pits
print("compute geodesic dist")
import time
t1=time.time()
g = ssg.add_edge_attribute_geodesic_distances_btw_pits(g,mesh)
t2=time.time()
print(t2-t1)

##### compute geodesic distance to the ridge
print("compute geodesic dist with potpouuri")
import time
t1=time.time()
g1 = ssg.add_edge_attribute_geodesic_distances_to_ridge_potpourri(g,mesh)
t2=time.time()
print(t2-t1)

print("compute geodesic dist to the ridge")
import time
t1=time.time()
g2 = ssg.add_edge_attribute_geodesic_distances_to_ridge(g,mesh)
t2=time.time()
print(t2-t1)

edges_euclidian_dist = (np.array(list(nx.get_edge_attributes(
            g, "euclidian_distance_btw_pits").values())))
edges_geo_dist = (np.array(list(nx.get_edge_attributes(
            g1, "geodesic_distance_btw_pits").values())))
edges_geo_pp_dist = (np.array(list(nx.get_edge_attributes(
            g2, "geodesic_distance_btw_pits").values())))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig_comp1 = go.Figure(data=[go.Scatter(
    x=edges_geo_dist,
    y=edges_euclidian_dist,
    mode='markers',
    marker_color='red',

)])
fig_comp1.show()

fig_comp2 = go.Figure(data=[go.Scatter(
    x=edges_geo_dist,
    y=edges_geo_pp_dist,
    mode='markers',
    marker_color='red',

)])
fig_comp2.show()

###############################################################################
# add mean value to nodes attributes
g = ssg.add_mean_value_to_nodes(g, dpf, attribute_name='basin_mean_depth')

###############################################################################
# get textures from graph
atex_labels, atex_pits, atex_ridges, atex_boundaries = ssg.get_textures_from_graph(g, mesh)

###############################################################################
# A more detailed computation of the sulcal graph
# with explicit call to the watershed
###############################################################################
# compute curvature, dpf and voronoi
mean_curvature, dpf, voronoi = swat.compute_mesh_features(mesh)

###############################################################################
# extract sulcal pits and associated basins
basins, ridges, adjacency = swat.watershed(
    mesh, voronoi, dpf, thresh_dist_perc, thresh_ridge, thresh_area_perc, mask)

# generate the sulcal graph
g = ssg.get_sulcal_graph(adjacency, basins, ridges)

###############################################################################
# generate the textures from graph
atex_labels_graph, atex_pits_graph, atex_ridges_graph, atex_boundaries = (ssg.get_textures_from_graph
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
print("Get the number of nodes in the graph:\n", g.number_of_nodes())
print("Get the number of edges in the graph:\n", g.number_of_edges())
print("Get the number of edges incident to the node 1:\n", g.degree[1])
print("Show the first node:\n", g.nodes[0])
print("Print the attributes of the first node:\n", g.nodes[0].keys())
print("Show the first edge:\n", g.edges[0,4])
print("Print the attributes of the first edge:\n", g.edges[0,4].keys())

###############################################################################
# Here is the way to get the list of 'interpolated_pits_index' for all nodes
interp_pits_inds = np.array(list(nx.get_node_attributes(g, 'interpolated_pits_index').values()))

###############################################################################
# add as a new node attribute the 3D coordinates of the vertex corresponding to
# # the inperpolated pit in the target spherical mesh
g = ssg.add_coords_attribute(g, target_spherical_mesh,
                                    attribute_vert_index='interpolated_pits_index',
                                    new_attribute_key='target_sphere_3dcoords')
print("Show the first node to see the impact of the previous command:\n", g.nodes[0])

###############################################################################
# add as a new node attribute the 3D coordinates of the vertex corresponding to
# # the inperpolated pit in the target mesh
g = ssg.add_coords_attribute(g, target_mesh,
                                    attribute_vert_index='interpolated_pits_index',
                                    new_attribute_key='target_mesh_3dcoords')
print("Show the first node to see the impact of the previous command:\n", g.nodes[0])

###############################################################################
# Get and visualize the distribution of some attirbutes on nodes and edges
nodes_pit_depth = (np.array(list(nx.get_node_attributes(
            g, "pit_depth").values())))
nodes_basin_area = (np.array(list(nx.get_node_attributes(
            g, "basin_area").values())))
nodes_basin_label = (np.array(list(nx.get_node_attributes(
            g, "basin_label").values())))

edges_ridge_depth = (np.array(list(nx.get_edge_attributes(
            g, "ridge_depth").values())))
edges_ridge_depth_min = (np.array(list(nx.get_edge_attributes(
            g, "ridge_depth_diff_min").values())))
edges_ridge_depth_max = (np.array(list(nx.get_edge_attributes(
            g, "ridge_depth_diff_max").values())))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=4)
## nodes_pit_depth
fig_nodes_pit_depth = go.Figure(data=[go.Histogram(
    x=nodes_pit_depth,
    nbinsx=10,  # Number of bins
    histnorm='probability',
    marker_color='blue',  # Bar color
    opacity=0.4  # Bar opacity
)])
# Update layout
fig_nodes_pit_depth.update_layout(
    title='nodes_pit_depth',
    xaxis_title='dpf*',
    yaxis_title='Count'
)
fig.add_trace(fig_nodes_pit_depth.data[0], row=1, col=1)
## pits depth
fig_nodes_basin_area = go.Figure(data=[go.Histogram(
    x=nodes_basin_area,
    nbinsx=10,  # Number of bins
    histnorm='probability',
    marker_color='green',  # Bar color
    opacity=0.4  # Bar opacity
)])
# Update layout
fig_nodes_basin_area.update_layout(
    title='nodes_basin_area',
    xaxis_title='mm2',
    yaxis_title='Count'
)
fig.add_trace(fig_nodes_basin_area.data[0], row=1, col=2)
## pits depth VS label
fig_nodes_pit_depth_vs_label = go.Figure(data=[go.Scatter(
    x=nodes_basin_label,
    y=nodes_pit_depth,
    mode='markers',
    marker_color='red',

)])
# Update layout
fig_nodes_pit_depth_vs_label.update_layout(
    title='pit_depth VS bassin_label',
    xaxis_title='label',
    yaxis_title='pit_depth'
)
fig.add_trace(fig_nodes_pit_depth_vs_label.data[0], row=1, col=3)
## basin_area VS label
fig_nodes_basin_area_vs_label = go.Figure(data=[go.Scatter(
    x=nodes_basin_label,
    y=nodes_basin_area,
    mode='markers',
    marker_color='black',
)])
# Update layout
fig_nodes_basin_area_vs_label.update_layout(
    title='basin_area VS bassin_label',
    xaxis_title='label',
    yaxis_title='basin_area'
)
fig.add_trace(fig_nodes_basin_area_vs_label.data[0], row=1, col=4)


### edges
## edges_ridge_depth
fig_edges_ridge_depth = go.Figure(data=[go.Histogram(
    x=edges_ridge_depth,
    nbinsx=10,  # Number of bins
    histnorm='probability',
    marker_color='yellow',  # Bar color
    opacity=0.4  # Bar opacity
)])
# Update layout
fig_edges_ridge_depth.update_layout(
    title='edges_ridge_depth',
    xaxis_title='dpf*',
    yaxis_title='Count'
)
fig.add_trace(fig_edges_ridge_depth.data[0], row=2, col=1)
## edges_ridge_depth_min
fig_edges_ridge_depth_min = go.Figure(data=[go.Histogram(
    x=edges_ridge_depth_min,
    nbinsx=10,  # Number of bins
    histnorm='probability',
    marker_color='cyan',  # Bar color
    opacity=0.4  # Bar opacity
)])
# Update layout
fig_edges_ridge_depth_min.update_layout(
    title='edges_ridge_depth_min',
    xaxis_title='dpf*',
    yaxis_title='Count'
)
fig.add_trace(fig_edges_ridge_depth_min.data[0], row=2, col=2)
## edges_ridge_depth_max
fig_edges_ridge_depth_max = go.Figure(data=[go.Histogram(
    x=edges_ridge_depth_max,
    nbinsx=10,  # Number of bins
    histnorm='probability',
    marker_color='violet',  # Bar color
    opacity=0.4  # Bar opacity
)])
# Update layout
fig_edges_ridge_depth_max.update_layout(
    title='edges_ridge_depth_max',
    xaxis_title='dpf*',
    yaxis_title='Count'
)
fig.add_trace(fig_edges_ridge_depth_max.data[0], row=2, col=3)
fig.show()

#############################################################################
# VISUALIZATION USING plotly
#############################################################################

import slam.plot as splt

mesh_data = {
    "vertices": mesh.vertices,
    "faces": mesh.faces,
    "opacity": 0.5,
    "title": 'Source'
}
intensity_data = {
    "values": dpf,
    "mode": "vertex",
}
fig1 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data)
fig1 = splt.plot_graph(g, coords_attribute='3dcoords', fig=fig1,
        marker={"size": 6, "color": "white", "line":{"color":"black", "width":2}},
)
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
trace_points = splt.plot_points(
    pits_coords,
    marker={"size": 6, "color": "white", "line":{"color":"black", "width":2}},
)
fig2.add_trace(trace_points)
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
pits_init = splt.plot_points(
    pits_coords,
    marker={"size": 6, "color": "white", "line":{"color":"black", "width":2}},
)
fig3.add_trace(pits_init)
# add the interpolated pits to the plot
interp_pits_coords_sphere = np.array(list(nx.get_node_attributes(g, 'target_sphere_3dcoords').values()))
pits_interpolated = splt.plot_points(
    interp_pits_coords_sphere,
    marker={"size": 6, "color": "red", "line":{"color":"black", "width":2}},
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
pits_interpolated = splt.plot_points(
    interp_pits_coords_mesh,
    marker={"size": 6, "color": "red", "line":{"color":"black", "width":2}}
)
fig4.add_trace(pits_interpolated)
fig4.show()
fig4



