"""
.. _example_screenshot_for_QC:

===================================
Examples of the use of plot for automatic screenshot e.g. for implementing visual quality check
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

mesh_data = {
    "vertices": mesh.vertices,
    "faces": mesh.faces,
    "title": "subject ID",  # figure title, default is None
}

fig1 = splt.plot_mesh(
    mesh_data=mesh_data,
    show_two_sides=True,
)
fig1.show()
fig1

###############################################################################
# To save the figure as png on the disc
# fig1.write_image("example_figure1.png", width=1300, height=900)

###############################################################################
# Example with a texture, here the DPF*

# Load a texture
tex = sio.load_texture("../examples/data/example_dpf_star.gii")

# set the parameters of the figure
intensity_data = {
    "values": tex.darray[0],
    "mode": "vertex",  # default is "cell"
    "cmin": -0.01,  # default is automatic value
    "cmax": 0.01,
}

display_settings = {
    "colorscale": "ylgnbu_r",  # color scale, default is Turbo
    "colorbar_label": "DPF*",  # colorbar label, default is None
}

fig2 = splt.plot_mesh(
    mesh_data,
    intensity_data,
    display_settings,
    show_two_sides=True,
)
fig2.show()
fig2

###############################################################################
# To save the figure as png on the disc
# fig2.write_image("example_figure2.png", width=1300, height=900)
# save the figure as an interactive HTML file
# fig1.write_html(SAVE_DIR)
