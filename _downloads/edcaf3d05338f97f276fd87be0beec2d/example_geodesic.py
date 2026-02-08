"""
.. _example_geodesic:

===================================
Geodesic in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Importation of slam modules
import slam.io as sio
import slam.geodesics as sgeo
import numpy as np
# import trimesh

###############################################################################
# Mesh importation
mesh = sio.load_mesh("../examples/data/example_mesh.gii")

###############################################################################
# Getting the index of vertices in specified geo_distance from a given vertex
vert_id = 528
max_geodist = 10
geo_distance = sgeo.compute_gdist(mesh, vert_id)
area_geodist_vi = np.where(geo_distance < max_geodist)[0]
print(area_geodist_vi)

###############################################################################
# For every vertex, get all the vertices within the maximum distance
area_geodist = sgeo.local_gdist_matrix(mesh, max_geodist)
print(area_geodist)

###############################################################################
# Get the index of the vertices located less than 10mm from another vertex
local_dist_vert_id = 644
vert_distmap = area_geodist[local_dist_vert_id].toarray()[0]
print(area_geodist)

###############################################################################
# Arbitrary indices of mesh.vertices to test with
start = 0
end = int(len(mesh.vertices) / 2.0)
path = sgeo.shortest_path(mesh, start, end)
print(path)

#############################################################################
# VISUALIZATION USING plotly
#############################################################################

import slam.plot as splt

display_settings = {}
display_settings['colorbar_label'] = 'Distance'
mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'Geodesic Distance'
intensity_data = {}
intensity_data['values'] = geo_distance
intensity_data["mode"] = "vertex"
fig1 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
fig1.show()
fig1

mesh_data['title'] = 'Local Geodesic Distance'
intensity_data['values'] = area_geodist[local_dist_vert_id].toarray().squeeze()
fig2 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
fig2.show()
fig2

###############################################################################
# # Visualization using pyglet
# mesh.visual.face_colors = [100, 100, 100, 100]
#
# ###############################################################################
# # Path3D with the path between the points
#
# path_visual = trimesh.load_path(mesh.vertices[path])
# path_visual

###############################################################################
# Visualizable two points
# points_visual = trimesh.points.PointCloud(mesh.vertices[[start, end]])

# ###############################################################################
# # Create a scene with the mesh, path, and points
#
# scene = trimesh.Scene([
#     points_visual,
#     path_visual,
#     mesh])
#
# scene.show(smooth=False)
