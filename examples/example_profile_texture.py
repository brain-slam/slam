"""
.. _example_profile_texture:

===================================
example of profile texture in slam
===================================
"""

# Authors: Tianqi SONG <tianqisong0117@gmail.com>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# # Visualization with visbrain

###############################################################################
# importation of slam modules
import numpy as np
import slam.surface_profiling as surfpf
import slam.io as sio

###############################################################################
# loading an example mesh and its texture
mesh_file = "../examples/data/example_mesh.gii"
texture_file = "../examples/data/example_texture.gii"
mesh = sio.load_mesh(mesh_file)
texture = sio.load_texture(texture_file)

###############################################################################
# Select a vertex and get the coordinate and normal.
vert_index = 1500
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
# Surface profiling, here we do the second round profiling
num_face = len(mesh.faces)
mesh_faces_id = np.linspace(0, num_face, num_face + 1).astype(int)
profile_samples, profile_co_faces = surfpf.second_round_profiling_vert(
    vert0,
    norm0,
    init_rot_dir,
    rot_angle,
    r_step,
    max_samples,
    mesh,
    mesh_faces_id
)

profile_points = profile_samples[:, :, 2]

###############################################################################
# Compute texture values of profile points
profile_samples = np.array([profile_samples])
profile_samples_fid = np.array([profile_co_faces])
profile_texture = surfpf.get_texture_value_on_profile(
    texture, mesh, profile_samples, profile_samples_fid
)

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# # Visualization with visbrain
# import trimesh.visual.color
# from matplotlib import pyplot as plt
# ###############################################################################
# # Generate colors of profile points
# # set color maps
# color_map = plt.get_cmap("jet", 12)
# # get texture value of profile points and mesh
# profile_tex = profile_texture.reshape(profile_texture.size)
# texture_value = texture.darray[0]
# # put them together to generate the colors
# prof_mesh_tex_color = np.hstack([profile_tex, texture_value])
# prof_mesh_points_color = trimesh.visual.color.interpolate(
#     prof_mesh_tex_color, color_map=color_map
# )
# mesh.visual.vertex_colors = prof_mesh_points_color[profile_texture.size:]
#
# # Create point cloud of profiling points
# prof_points_mesh = profile_points.reshape(
#     profile_points.shape[0] * profile_points.shape[1], 3
# )
# prof_points_colors = prof_mesh_points_color[: profile_texture.size]
# points_mesh = trimesh.points.PointCloud(
#     prof_points_mesh, colors=np.array(prof_points_colors, dtype=np.uint8)
# )
#
# scene = trimesh.Scene([points_mesh, mesh])
# scene.show(smooth=False)
