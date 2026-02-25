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

interpolated_tex_values = srem.texture_spherical_interpolation_nearest_neighbor(
    source_spherical_mesh, target_spherical_mesh, source_tex.darray[0], normalize_spheres=True
)

#############################################################################
# VISUALIZATION USING plotly
#############################################################################

import slam.plot as splt

# Plot Mean Curvature on the source mesh
mesh_data = {
    "vertices": source_mesh.vertices,
    "faces": source_mesh.faces,
    "title": 'Source'
}
intensity_data = {
    "values": source_tex.darray[0],
    "mode": "vertex",
}
fig1 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data)
fig1.show()
fig1

# Plot Mean Curvature on the spherical mesh
mesh_data = {
    "vertices": source_spherical_mesh.vertices,
    "faces": source_spherical_mesh.faces,
    "title": 'Spherical Source'
}

fig2 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data)
source_vert = splt.create_hover_trace(
    source_spherical_mesh.vertices,
    marker={"size": 4, "color": "black"},
)
target_vert = splt.create_hover_trace(
    target_spherical_mesh.vertices,
    marker={"size": 4, "color": "white"},
)
fig2.add_trace(source_vert)
fig2.add_trace(target_vert)
fig2.show()
fig2

# Plot Mean Curvature
mesh_data = {
    "vertices": target_mesh.vertices,
    "faces": target_mesh.faces,
    "title": 'target mesh with curvature from source mesh'
}
intensity_data = {
    "values": interpolated_tex_values,
    "mode": "vertex",
}
fig3 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data)
fig3.show()
fig3
