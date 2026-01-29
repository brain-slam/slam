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
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

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
# Getting the vertex index in specified geo_distance of vert
vert_id = 200
max_geodist = 10
geo_distance = sgeo.compute_gdist(mesh, vert_id)
area_geodist_vi = np.where(geo_distance < max_geodist)[0]
print(area_geodist_vi)

###############################################################################
# Getting the vertex index in specified geo_distance of vert
area_geodist = sgeo.local_gdist_matrix(mesh, max_geodist)

###############################################################################
# Get the vertex index
indices = []

for i in range(mesh.vertices.shape[0]):
    vert_distmap = area_geodist[i].toarray()[0]
    area_geodist_v = np.where(vert_distmap > 0)[0]
    indices += [area_geodist_v]

###############################################################################
# Arbitrary indices of mesh.vertices to test with
start = 0
end = int(len(mesh.vertices) / 2.0)
path = sgeo.shortest_path(mesh, start, end)
print(path)

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# # Visualization with visbrain
# # show the geodesic distance
# visb_sc = splt.visbrain_plot(
#     mesh=mesh,
#     tex=geo_distance,
#     caption="geodesic distance",
#     cblabel="distance"
# )
# visb_sc.preview()
# # show the geodesic distance locally
# visb_sc2 = splt.visbrain_plot(
#     mesh=mesh,
#     tex=area_geodist[0].toarray().squeeze(),
#     caption="local geodesic distance",
#     cblabel="distance",
# )
# visb_sc2.preview()

###############################################################################
# # Visualization using pyglet
#
# mesh.visual.face_colors = [100, 100, 100, 100]
#
# ###############################################################################
# # Path3D with the path between the points
#
# path_visual = trimesh.load_path(mesh.vertices[path])
# path_visual
#
# ###############################################################################
# # Visualizable two points
# points_visual = trimesh.points.PointCloud(mesh.vertices[[start, end]])
#
# ###############################################################################
# # Create a scene with the mesh, path, and points
#
# scene = trimesh.Scene([
#     points_visual,
#     path_visual,
#     mesh])
#
# scene.show(smooth=False)
