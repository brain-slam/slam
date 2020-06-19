"""
.. _example_geodesic:

===================================
Geodesic in slam
===================================
"""

# Authors: Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Importation of slam modules
import slam.plot as splt
import slam.io as sio
import slam.geodesics as sgeo
import numpy as np
import trimesh

###############################################################################
# Mesh importation

mesh = sio.load_mesh('data/example_mesh.gii')

###############################################################################
# Getting the vertex index in specified geo_distance of vert

vert_id = 0
max_geodist = 10

geo_distance = sgeo.compute_gdist(mesh, vert_id)
area_geodist_vi = np.where(geo_distance < max_geodist)[0]

area_geodist_vi

###############################################################################
# Visualization

visb_sc = splt.visbrain_plot(mesh=mesh, tex=geo_distance,
                             caption='geodesic distance',
                             cblabel='distance')
visb_sc.preview()

###############################################################################
# Getting the vertex index in specified geo_distance of vert


area_geodist = sgeo.local_gdist_matrix(mesh, max_geodist)

visb_sc = splt.visbrain_plot(mesh=mesh,
                             tex=area_geodist[0].toarray().squeeze(),
                             caption='local geodesic distance',
                             cblabel='distance', visb_sc=visb_sc)
visb_sc.preview()
###############################################################################
# Print the vertex index

for i in range(mesh.vertices.shape[0]):
    vert_distmap = area_geodist[i].toarray()[0]
    area_geodist_v = np.where(vert_distmap > 0)[0]
    print(area_geodist_v)

###############################################################################
# Arbitrary indices of mesh.vertices to test with

start = 0
end = int(len(mesh.vertices) / 2.0)
path = sgeo.shortest_path(mesh, start, end)
path



###############################################################################
# Visualization

mesh.visual.face_colors = [100, 100, 100, 100]

###############################################################################
# Path3D with the path between the points

path_visual = trimesh.load_path(mesh.vertices[path])
path_visual

###############################################################################
# visualizable two points
points_visual = trimesh.points.PointCloud(mesh.vertices[[start, end]])

###############################################################################
# create a scene with the mesh, path, and points

scene = trimesh.Scene([
    points_visual,
    path_visual,
    mesh])

scene.show(smooth=False)
