"""
.. _example_remeshing:

===================================
Remeshing example in slam
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
import slam.remeshing as srem
import numpy as np
###############################################################################
# Source object files
source_mesh_file = "../examples/data/example_mesh.gii"
source_texture_file = "../examples/data/example_texture.gii"
source_spherical_mesh_file = "../examples/data/example_mesh_spherical.gii"

###############################################################################
# Target object files
target_mesh_file = "../examples/data/example_mesh_2.gii"
target_spherical_mesh_file = "../examples/data/example_mesh_2_spherical.gii"

source_mesh = sio.load_mesh(source_mesh_file)
source_tex = sio.load_texture(source_texture_file)
source_spherical_mesh = sio.load_mesh(source_spherical_mesh_file)

target_mesh = sio.load_mesh(target_mesh_file)
target_spherical_mesh = sio.load_mesh(target_spherical_mesh_file)

interpolated_tex_values = srem.spherical_interpolation_nearest_neigbhor(
    source_spherical_mesh, target_spherical_mesh, source_tex.darray[0]
)

#############################################################################
# VISUALIZATION USING INTERNAL TOOLS
#############################################################################

import slam.plot as splt

source_mesh.apply_transform(source_mesh.principal_inertia_transform)
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, source_mesh.vertices.T).T

# Plot Mean Curvature
display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = source_mesh.faces
mesh_data['title'] = 'Source'
intensity_data = {}
intensity_data['values'] = source_tex.darray[0],
intensity_data["mode"] = "vertex"
Fig = splt.mesh_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
# Fig.show()
Fig.write_image("example_distorsion_1.png")


# # Visualization with visbrain
# import slam.plot as splt
# ###############################################################################

source_spherical_mesh.apply_transform(
    source_spherical_mesh.principal_inertia_transform)
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, source_spherical_mesh.vertices.T).T

# Plot Mean Curvature
display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = source_spherical_mesh.faces
mesh_data['title'] = 'Spherical Source'
intensity_data = {}
intensity_data['values'] = source_tex.darray[0],
intensity_data["mode"] = "vertex"
Fig = splt.mesh_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
# Fig.show()
Fig.write_image("example_remeshing_1.png")

target_mesh.apply_transform(target_mesh.principal_inertia_transform)
theta = np.pi / 2
rot_x = np.array([[1, 0, 0],
                  [0, np.cos(theta), -np.sin(theta)],
                  [0, np.sin(theta),  np.cos(theta)]])
vertices_translate = np.dot(rot_x, target_mesh.vertices.T).T

# Plot Mean Curvature
display_settings = {}
mesh_data = {}
mesh_data['vertices'] = vertices_translate
mesh_data['faces'] = target_mesh.faces
mesh_data['title'] = 'target mesh from source mesh'
intensity_data = {}
intensity_data['values'] = interpolated_tex_values,
intensity_data["mode"] = "vertex"
Fig = splt.mesh_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
# Fig.show()
Fig.write_image("example_remeshing_2.png")
