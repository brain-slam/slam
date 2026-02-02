"""
.. _example_registration:

===================================
Show mesh registration features in slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# importation of slam modules
import slam.io as sio
import trimesh
import numpy as np
###############################################################################
# loading an two unregistered meshes
mesh_1 = sio.load_mesh("../examples/data/example_mesh.gii")
mesh_2 = sio.load_mesh("../examples/data/example_mesh_2.gii")

###############################################################################
# compute ICP registration
# this functionnality requires to install the optional package rtree
transf_mat, cost = trimesh.registration.mesh_other(
    mesh_1, mesh_2, samples=500, scale=False, icp_first=10, icp_final=100
)
print(transf_mat)
print(cost)

#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################

# # plot them to check the misalignment
joint_mesh = mesh_1 + mesh_2
joint_tex = np.ones((joint_mesh.vertices.shape[0],))
joint_tex[: mesh_1.vertices.shape[0]] = 10

import slam.plot as splt

joint_mesh.apply_transform(joint_mesh.principal_inertia_transform)
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, joint_mesh.vertices.T).T

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = joint_mesh.faces
mesh_data['title'] = 'Before Registration'
intensity_data = {}
intensity_data['values'] = joint_tex
intensity_data["mode"] = "vertex"
Fig = splt.mesh_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
# Fig.show()
Fig.write_image("example_registration_1.png")

###############################################################################
# apply the estimated rigid transformation to the mesh
mesh_1.apply_transform(transf_mat)
joint_mesh = mesh_1 + mesh_2
joint_tex = np.ones((joint_mesh.vertices.shape[0],))
joint_tex[: mesh_1.vertices.shape[0]] = 10
joint_mesh.apply_transform(joint_mesh.principal_inertia_transform)
vertices_translate = np.dot(rot_x, joint_mesh.vertices.T).T

display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = joint_mesh.faces
mesh_data['title'] = 'After Registration'
intensity_data = {}
intensity_data['values'] = joint_tex
intensity_data["mode"] = "vertex"
Fig = splt.mesh_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
# Fig.show()
Fig.write_image("example_registration_2.png")
