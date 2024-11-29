"""
.. _example_topology:

===================================
Show topology manipulation tools in slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# importation of slam modules
import slam.io as sio
import slam.topology as stop
import slam.generate_parametric_surfaces as sps
import numpy as np


###############################################################################
# here is how to get the vertices that define the boundary of an open mesh
K = [-1, -1]
open_mesh = sps.generate_quadric(K, nstep=[5, 5])

###############################################################################
# Identify the vertices lying on the boundary of the mesh and order
# them to get a path traveling across boundary vertices
# The output is a list of potentially more than one boudaries
# depending on the topology of the input mesh.
# Here the mesh has a single boundary
open_mesh_boundary = stop.mesh_boundary(open_mesh)
print(open_mesh_boundary)

###############################################################################
# erode the mesh by removing the faces having 3 vertices on the boundary
eroded_mesh = stop.remove_mesh_boundary_faces(open_mesh, face_vertex_number=1)

###############################################################################
# here is how to get the vertices that define the boundary of
# a texture on a mesh
# Let us first load example data
mesh = sio.load_mesh("../examples/data/example_mesh.gii")
# rotate the mesh for better visualization
transfo_flip = np.array(
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
mesh.apply_transform(transfo_flip)

# Load the example texture and compute its boundary
tex_parcel = sio.load_texture("../examples/data/example_texture_parcel.gii")
texture_bound = stop.texture_boundary(mesh, tex_parcel.darray[0], 20)

###############################################################################
# ================= cut_mesh =================
# Cut he mesh into subparts corresponding to the different values in
# the texture tex_parcel
parc_u = np.unique(tex_parcel.darray[0])
print(
    "Here the texture contains {0} different values: {1}" "".format(
        len(parc_u), parc_u)
)

###############################################################################
# Let us cut the mesh
sub_meshes, sub_tex, sub_corresp = stop.cut_mesh(mesh, tex_parcel.darray[0])
# The output submeshes are ordered following texture values
# The second output of cut_mesh gives the texture value corresponding to
# each submesh
print(
    "Corresponding texture values are given by" " the second ouput: {}".format(
        sub_tex)
)

# The respective indices of the vertices of each submesh in the original
# mesh before the cut is given by the third output:
print(sub_corresp)

###############################################################################
# ================= close_mesh =================
# close the largest submesh
cuted_mesh = sub_meshes[-1]
mesh_closed, nb_verts_added = stop.close_mesh(cuted_mesh)

# The closed mesh is watertight while before closing if was not
print(mesh.is_watertight)
print(mesh_closed.is_watertight)

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# import slam.plot as splt
# from vispy.scene import Line
# from visbrain.objects import VispyObj, SourceObj
#
# ###############################################################################
# # show the result
# # WARNING : BrainObj should be added first before
# visb_sc = splt.visbrain_plot(mesh=open_mesh, caption="open mesh")
# # create points with vispy
# for bound in open_mesh_boundary:
#     points = open_mesh.vertices[bound]
#     s_rad = SourceObj(
#         "rad",
#         points,
#         color="red",
#         symbol="square",
#         radius_min=10)
#     visb_sc.add_to_subplot(s_rad)
#     lines = Line(pos=open_mesh.vertices[bound], width=10, color="b")
#     # wrap the vispy object using visbrain
#     l_obj = VispyObj("line", lines)
#     visb_sc.add_to_subplot(l_obj)
# visb_sc.preview()
#
# ###############################################################################
# # show the result
# visb_sc2 = splt.visbrain_plot(mesh=eroded_mesh, caption="eroded mesh")
# # show again the boundary of original mesh which have been removed with
# # corresponding faces by the erosion
# for bound in open_mesh_boundary:
#     points = open_mesh.vertices[bound]
#     s_rad = SourceObj(
#         "rad",
#         points,
#         color="red",
#         symbol="square",
#         radius_min=10)
#     visb_sc2.add_to_subplot(s_rad)
#     lines = Line(pos=open_mesh.vertices[bound], width=10, color="b")
#     # wrap the vispy object using visbrain
#     l_obj = VispyObj("line", lines)
#     visb_sc2.add_to_subplot(l_obj)
# visb_sc2.preview()
#
# ###############################################################################
# # show the results
# visb_sc3 = splt.visbrain_plot(
#     mesh=mesh, tex=tex_parcel.darray[0], caption="texture boundary"
# )
# cols = ["red", "green", "yellow", "blue"]
# ind = 0
# for bound in texture_bound:
#     points = mesh.vertices[bound]
#     s_rad = SourceObj(
#         "rad",
#         points,
#         color="red",
#         symbol="square",
#         radius_min=10)
#     visb_sc3.add_to_subplot(s_rad)
#     lines = Line(pos=mesh.vertices[bound], width=10, color=cols[ind])
#     # wrap the vispy object using visbrain
#     l_obj = VispyObj("line", lines)
#     visb_sc3.add_to_subplot(l_obj)
#     ind += 1
#     if ind == len(cols):
#         ind = 0
# visb_sc3.preview()
#
# ###############################################################################
# # show the mesh with the cuted subparts in different colors
# scene_list = list()
# cuted_mesh = sub_meshes[-1]
# joint_mesh = sub_meshes[0]
# joint_tex = np.zeros((sub_meshes[0].vertices.shape[0],))
# last_ind = sub_meshes[0].vertices.shape[0]
# for ind, sub_mesh in enumerate(sub_meshes[1:]):
#     sub_tex = np.ones((sub_mesh.vertices.shape[0],)) * (ind + 1)
#     joint_mesh += sub_mesh
#     joint_tex = np.hstack((joint_tex, sub_tex))
# visb_sc3 = splt.visbrain_plot(
#     mesh=joint_mesh,
#     tex=joint_tex,
#     caption="mesh parts shown in different colors"
# )
# ind = 0
# boundaries = stop.mesh_boundary(cuted_mesh)
# for bound in boundaries:
#     s_rad = SourceObj(
#         "rad",
#         cuted_mesh.vertices[bound],
#         color=cols[ind],
#         symbol="square",
#         radius_min=10,
#     )
#     visb_sc3.add_to_subplot(s_rad)
#     lines = Line(pos=cuted_mesh.vertices[bound], color=cols[ind], width=10)
#     # wrap the vispy object using visbrain
#     l_obj = VispyObj("line", lines)
#     visb_sc3.add_to_subplot(l_obj)
#     ind += 1
#     if ind == len(cols):
#         ind = 0
# visb_sc3.preview()
# ###############################################################################
# # show the largest submesh with the boundaries of cutted parts
# visb_sc4 = splt.visbrain_plot(mesh=cuted_mesh, caption="open mesh")
# # create points with vispy
# for bound in boundaries:
#     points = cuted_mesh.vertices[bound]
#     s_rad = SourceObj(
#         "rad",
#         points,
#         color="blue",
#         symbol="square",
#         radius_min=10)
#     visb_sc4.add_to_subplot(s_rad)
#     lines = Line(pos=cuted_mesh.vertices[bound], width=10, color="r")
#     # wrap the vispy object using visbrain
#     l_obj = VispyObj("line", lines)
#     visb_sc4.add_to_subplot(l_obj)
# visb_sc4.preview()
# ###############################################################################
# # show the closed mesh
# visb_sc5 = splt.visbrain_plot(mesh=mesh_closed, caption="closed mesh")
# visb_sc5.preview()
