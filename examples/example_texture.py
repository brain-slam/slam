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
from slam import plot as plt

###############################################################################
#
tex = sio.load_texture("examples/data/example_texture.gii")
mesh = sio.load_mesh("examples/data/example_mesh.gii")
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
print('extremum texture')
mesh = sio.load_mesh("../examples/data/example_mesh.gii")
print('maximum')
print(np.count_nonzero(tex.extremum(mesh) == 1))
print('minimum')
print(np.count_nonzero(tex.extremum(mesh) == -1))

###############################################################################
# plot

# dict for proj
NAME_TEX = "sulc"
TITLE = "test"
EXT = "png"
PATH = Path("./test")
SAVE_DIR = PATH / f"{TITLE}.{EXT}"

mesh_data = {
    "vertices": mesh.vertices,
    "faces": mesh.faces,
    "center": mesh.center_mass,
    "title": TITLE
}
intensity_data = {"values": tex.darray[0], "mode": "vertex"}
display_settings = {"colorscale": "Turbo", "colorbar_label": NAME_TEX}

fig = plt.mes3d_projection(
    mesh_data,
    intensity_data,
    display_settings,
)

os.makedirs(PATH, exist_ok=True)
fig.write_image(SAVE_DIR, width=1600, height=900)
