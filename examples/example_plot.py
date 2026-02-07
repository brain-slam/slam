"""
.. _example_plot:

===================================
Examples of the use of plot in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# importation of slam modules
import slam.io as sio
import slam.plot as splt

###############################################################################
# Load a mesh
mesh_file = "../examples/data/example_mesh.gii"
mesh = sio.load_mesh(mesh_file)

mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'Simplest plot'
Fig = splt.plot_mesh(
    mesh_data=mesh_data)
Fig.show()

###############################################################################
# To save the figure as png on the disc
# Fig.write_image("example_figure.png")

###############################################################################
# Load a texture
tex = sio.load_texture("../examples/data/example_dpf.gii")
# To tune the visualization of the texture, use the "intensity_data" dict:
#         "intensity": intensity_data["values"],
#         "intensitymode": intensity_data.get("mode", "cell"),
#         "colorscale": display_settings.get("colorscale", "Turbo"),
#         "cmin": intensity_data.get("cmin", None),
#         "cmax": intensity_data.get("cmax", None),

# The "display_settings" dict can also be used:
#         "colorbar": {
#             "title": display_settings.get("colorbar_label", ""),
#         "colorbar_tickvals": display_settings.get("tickvals", None),
#         "colorbar_ticktext": display_settings.get("ticktext", None),

mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'Plot a texture on a mesh'
intensity_data = {}
intensity_data['values'] = tex.darray[0]
intensity_data["mode"] = "vertex"
display_settings = {}
display_settings['colorbar_label'] = 'DPF'
Fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()

