"""
.. _example_surface_profiling:

===================================
example of surface profiling in slam
===================================
"""

# Authors: Tianqi SONG <tianqisong0117@gmail.com>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# importation of slam modules
import numpy as np
import slam.surface_profiling as surfpf
import slam.io as sio

###############################################################################
# loading an example mesh
mesh = sio.load_mesh("../examples/data/example_mesh.gii")

###############################################################################
# Select a vertex and get the coordinate and normal.
vert_index = 1000
vert0 = mesh.vertices[vert_index]
norm0 = mesh.vertex_normals[vert_index]

###############################################################################
# Set the parameters for surface profiling
# initial direction of rotation, rotation angle, length and number of
# sampling steps
init_rot_dir = np.array([1, 1, 1]) - vert0
rot_angle = 10
r_step = 0.1
max_samples = 45

###############################################################################
# Surface profiling
profile_points = surfpf.surface_profiling_vert(
    vert0, norm0, init_rot_dir, rot_angle, r_step, max_samples, mesh
)

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# import trimesh.visual.color
# ###############################################################################
# # Visualize result
# prof_points_mesh = profile_points.reshape(
#     profile_points.shape[0] * profile_points.shape[1], 3
# )
# prof_points_colors = np.zeros(prof_points_mesh.shape)
# points_mesh = trimesh.points.PointCloud(
#     prof_points_mesh, colors=np.array(prof_points_colors, dtype=np.uint8)
# )
# # set the points colors
# red, yellow, green, blue = [
#     [255, 0, 0, 255],
#     [255, 255, 0, 255],
#     [0, 255, 0, 255],
#     [0, 0, 255, 255],
# ]
# color_i = np.zeros(4)
# for i in range(len(prof_points_mesh)):
#     degree = i / max_samples
#     num_profiles = int(360 / rot_angle)
#     segment_color = num_profiles / 3
#
#     if degree <= segment_color:
#         color_i = trimesh.visual.color.linear_color_map(
#             degree / segment_color, [red, yellow]
#         )
#     elif segment_color < degree <= segment_color * 2:
#         color_i = trimesh.visual.color.linear_color_map(
#             degree / segment_color - 1, [yellow, green]
#         )
#     elif segment_color * 2 < degree <= num_profiles:
#         color_i = trimesh.visual.color.linear_color_map(
#             degree / segment_color - 2, [green, blue]
#         )
#     points_mesh.colors[i] = np.array(color_i)
#
# # create a scene with the mesh and profiling points
# scene = trimesh.Scene([points_mesh, mesh])
# scene.show(smooth=False)
