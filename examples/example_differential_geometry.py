"""
.. _example_differential_geometry:

===================================
example of differential geometry tools in slam
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
import slam.differential_geometry as sdg
import numpy as np
###############################################################################
# loading an examplar mesh and corresponding texture
mesh_file = "../examples/data/example_mesh.gii"
texture_file = "../examples/data/example_texture.gii"
mesh = sio.load_mesh(mesh_file)
tex = sio.load_texture(texture_file)

###############################################################################
# compute various types of Laplacian of the mesh
lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type="fem")
print(mesh.vertices.shape)
print(lap.shape)
lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type="conformal")
lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type="meanvalue")
lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type="authalic")

###############################################################################
# smooth the mesh using Laplacian
s_mesh = sdg.laplacian_mesh_smoothing(mesh, nb_iter=100, dt=0.1)

###############################################################################
# compute the gradient of texture tex
triangle_grad = sdg.triangle_gradient(mesh, tex.darray[0])
print(triangle_grad)
grad = sdg.gradient(mesh, tex.darray[0])
print(grad)
norm_grad = sdg.norm_gradient(mesh, tex.darray[0])
print(norm_grad)

#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################

import slam.plot as splt

vertices = mesh.vertices
# center the vertices
vertices = vertices - np.mean(vertices, axis=0)
vertices_translate = np.copy(vertices)
# rotate the vertices
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, vertices_translate.T).T
rot_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1], ])
vertices_translate = np.dot(rot_z, vertices_translate.T).T

# Plot Original Mesh
display_settings = {}
display_settings['colorbar_label'] = 'Curvature'
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'example_mesh.gii Mean Curvature'
intensity_data = {}
intensity_data['values'] = tex.darray[0]
intensity_data["mode"] = "vertex"
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()


# Show the Norm of the Gradient of the Curvature
display_settings = {}
display_settings['colorbar_label'] = 'Gradient Magnitude'
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'example_mesh.gii Norm of the Gradient of Curvature'
intensity_data = {}
intensity_data['values'] = norm_grad
intensity_data["mode"] = "vertex"
Fig = splt.mes3d_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

