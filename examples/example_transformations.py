"""
.. _example_transfo_mesh:

===================================
Transformation of mesh example in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Maxime Van Der Valk <maxime.vandervalk@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# importation of slam modules
import slam.io as sio
import numpy as np
import trimesh.transformations as ttrans

###############################################################################
# Load a mesh
mesh_file = "../examples/data/example_mesh.gii"
mesh = sio.load_mesh(mesh_file)

###############################################################################
# Apply a random transformation to the mesh
mesh.apply_transform(ttrans.random_rotation_matrix())

###############################################################################
# reortient the mesh according to its principal inertia axes
mesh.apply_transform(mesh.principal_inertia_transform)

###############################################################################
# Define and apply a rotation around the x axis
theta = np.pi / 6
rotation_x = np.array([[1, 0, 0, 0],[0, np.cos(theta), -np.sin(theta), 0],[0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])
mesh.apply_transform(rotation_x)


#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################
import slam.plot as splt

mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'Example Transformations'
Fig = splt.plot_mesh(
    mesh_data=mesh_data)
Fig.show()

