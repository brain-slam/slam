"""
.. _example_sulcal_depth:

===================================
example of sulcal depth estimation in slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# importation of slam modules
import slam.io as sio
import slam.sulcal_depth as sdepth

###############################################################################
# loading an examplar mesh and corresponding texture
mesh_file = "../examples/data/example_mesh.gii"
texture_file = "../examples/data/example_texture.gii"
mesh = sio.load_mesh(mesh_file)

###############################################################################
# compute the depth potential function
dpf = sdepth.depth_potential_function(mesh)

###############################################################################
# compute the dpf_star
dpf_star = sdepth.dpf_star(mesh)


#############################################################################
# VISUALIZATION USING plotly
#############################################################################

import slam.plot as splt

display_settings = {}
display_settings['colorbar_label'] = 'dpf_star'
mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'dpf_star'
intensity_data = {}
intensity_data['values'] = dpf_star[0]
intensity_data["mode"] = "vertex"
fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
fig.show()
fig
