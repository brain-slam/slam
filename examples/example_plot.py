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
fig1 = splt.plot_mesh(
    mesh_data=mesh_data)
fig1.show()
fig1

###############################################################################
# To save the figure as png on the disc
# fig1.write_image("example_figure.png", width=1600, height=900)
# save the figure as an interactive HTML file
# fig1.write_html(SAVE_DIR)

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
fig2 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
fig2.show()
fig2


###############################################################################
# Complete example with all modifiable parameters
mesh_data = {
    "vertices": mesh.vertices,
    "faces": mesh.faces,
    "title": "the title",  # figure title, default is None
}

intensity_data = {
    "values": tex.darray[0],
    "mode": "vertex",  # default is "cell"
    "cmin": 0,  # default is automatic value
    "cmax": 1,
}

display_settings = {
    "colorscale": "RdBu",  # color scale, default is Turbo
    "colorbar_label": "name of the texture",  # colorbar label, default is None
    "template": "plotly_dark",  # default is blank theme without axes
    "tickvals": [
        0,
        0.25,
        0.5,
        0.75,
        1,
    ],  # default is None, sets exact tick positions on the colorbar
    "ticktext": [
        "0%",
        "25%",
        "50%",
        "75%",
        "100%",
    ],  # default is None, customizes tick labels on the colorbar
}

fig3 = splt.plot_mesh(
    mesh_data,
    intensity_data,
    display_settings,
    caption=True,  # snapshot, default is None
)

# add an additional trace
# example: display vertex index on hover and customize vertex color and size
# over the mesh
hover_text = [f"vertex {i}" for i in range(len(mesh.vertices))]
trace_hover = splt.create_hover_trace(
    mesh.vertices,
    text=hover_text,
    marker={"size": 4, "color": "blue"},
)

fig3.add_trace(trace_hover, row=1, col=1)
fig3.add_trace(trace_hover, row=1, col=2)
fig3.show()
fig3

