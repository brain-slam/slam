"""
.. _example_registration:

===================================
Show mesh registration features in slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# importation of slam modules
import slam.plot as splt
import slam.io as sio
import numpy as np
import trimesh

###############################################################################
# loading an two unregistered meshes
mesh_1 = sio.load_mesh('../examples/data/example_mesh.gii')
mesh_2 = sio.load_mesh('../examples/data/example_mesh_2.gii')

###############################################################################
# plot them to check the misalignment
joint_mesh = mesh_1 + mesh_2
joint_tex = np.ones((joint_mesh.vertices.shape[0],))
joint_tex[:mesh_1.vertices.shape[0]] = 10
visb_sc = splt.visbrain_plot(mesh=joint_mesh, tex=joint_tex,
                             caption='before registration')
visb_sc.preview()

###############################################################################
# compute ICP registration
transf_mat, cost = trimesh.registration.mesh_other(mesh_1, mesh_2,
                                                   samples=500,
                                                   scale=False,
                                                   icp_first=10,
                                                   icp_final=100)
print(transf_mat)
print(cost)

###############################################################################
# apply the estimated rigid transformation to the mesh
mesh_1.apply_transform(transf_mat)

###############################################################################
# plot the two meshes to check they are now aligned
joint_mesh = mesh_1 + mesh_2
visb_sc = splt.visbrain_plot(mesh=joint_mesh, tex=joint_tex,
                             caption='after registration')
visb_sc.preview()
