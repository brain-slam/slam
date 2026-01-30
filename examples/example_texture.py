"""
.. _example_texture:

===================================
Texture example in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Importation of slam modules
import os
from pathlib import Path
import numpy as np
from slam import texture
from slam import io as sio
from slam import plot as proj

###############################################################################
#
tex = sio.load_texture("../examples/data/example_dpf.gii")
mesh = sio.load_mesh("../examples/data/example_mesh.gii")
print(tex)
print(tex.metadata)
print(tex.shape)
print(tex.dtype)
print(tex.min())
print(tex.max())


###############################################################################
#
darray = np.zeros((2, 3))
tex2 = texture.TextureND(darray=darray)
print(tex2.metadata)
print(tex2)
print(tex2.shape)
print(tex2.dtype)
print(tex2.min())
print(tex2.max())
sio.write_texture(tex2, "test.gii")

#############
print("extremum texture")
print("maximum")
print(np.count_nonzero(tex.extremum(mesh) == 1))
print("minimum")
print(np.count_nonzero(tex.extremum(mesh) == -1))

###############################################################################
# plot

# reorient the mesh
mesh.apply_transform(mesh.principal_inertia_transform)
theta = np.pi / 2
rot_x = np.array(
    [[1, 0, 0],
     [0, np.cos(theta), -np.sin(theta)],
     [0, np.sin(theta), np.cos(theta)]]
)
vertices_translate = np.dot(rot_x, mesh.vertices.T).T

# parameters for the projection
# Complete example with all modifiable parameters
NAME_TEX = "sulc"
TITLE = "test"
EXT = "html"
PATH = Path("./test")
SAVE_DIR = PATH / f"{TITLE}.{EXT}"

data = tex.darray[0]

mesh_data = {
    "vertices": vertices_translate,
    "faces": mesh.faces,
    "title": TITLE,  # figure title, default is None
}

intensity_data = {
    "values": data,
    "mode": "vertex",  # default is "cell"
    "cmin": 0,  # default is automatic value
    "cmax": 1,
}

display_settings = {
    "colorscale": "RdBu",  # color scale, default is Turbo
    "colorbar_label": NAME_TEX,  # colorbar label, default is None
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

fig = proj.mesh_projection(
    mesh_data,
    intensity_data,
    display_settings,
)

# add an additional trace
# example: display vertex index on hover and customize vertex color and size
# over the mesh
hover_text = [f"vertex {i}" for i in range(len(vertices_translate))]
trace_hover = proj.create_hover_trace(
    vertices_translate,
    text=hover_text,
    marker={"size": 4, "color": "blue"},
)

fig.add_trace(trace_hover)

# save the figure as an HTML file
# os.makedirs(PATH, exist_ok=True)
# fig.write_html(SAVE_DIR)
