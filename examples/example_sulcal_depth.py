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
# This is a tutorial for computing sulcal depth using the method called DPF*
# introduced in the publication:
# Dieudonné, M., Auzias, G., Lefèvre, J., "Scale-invariant brain morphometry:
# application to sulcal depth", 2026, https://arxiv.org/abs/2501.05436


###############################################################################
# import numpy
import numpy as np
# importation of slam modules
import slam.io as sio
import slam.sulcal_depth as sdepth

###############################################################################
# loading an examplar mesh and corresponding texture
mesh_file = "../examples/data/example_mesh.gii"
texture_file = "../examples/data/example_texture.gii"
mesh = sio.load_mesh(mesh_file)

###############################################################################
# The dpf* can be computed using a single function.
# The outuput of this function are
# dpf_star, a texture (vetcor of size equal to the number of vertices in the mesh)
# corresponding to the normalized sulcal depth;
# lc, the estimated characteristic length of the mesh
dpf_star, lc = sdepth.dpf_star(mesh)

###############################################################################
# The characteristic length can then be used to compute the absolute DPF*:
print("Estimated characteristic length", lc)
abs_dpf_star = lc * dpf_star[0]

###############################################################################
# Compare the two with simple stats
print("mean(dpf_star)=", np.mean(dpf_star[0]))
print("std(dpf_star)=", np.std(dpf_star))
print("mean(abs_dpf_star)=", np.mean(abs_dpf_star))
print("std(abs_dpf_star)=", np.std(abs_dpf_star))

###############################################################################
# See the article following the link provided at the top of this example for
# more details about the method and its potential applications.

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
