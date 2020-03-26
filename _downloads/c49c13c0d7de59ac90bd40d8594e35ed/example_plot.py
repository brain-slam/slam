"""
.. _example_plot:

===================================
Show basic plot in slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# This script shows examples of visualization in SLAM.
# The visu is based on visbrain: https://github.com/EtienneCmb/visbrain

import slam.plot as splt
import slam.io as sio
import numpy as np

# loading an examplar mesh and corresponding texture
mesh_file = '../examples/data/example_mesh.gii'
texture_file = '../examples/data/example_texture.gii'
mesh = sio.load_mesh(mesh_file)
tex = sio.load_texture(texture_file)

###############################################################################
# here is the range of values in the texture:
print('[{a:2.3f}, {b:2.3f}]'.format(a=tex.min(), b=tex.max()))

###############################################################################
# make the figure
visb_sc = splt.visbrain_plot(mesh=mesh, caption='simple mesh')
visb_sc = splt.visbrain_plot(mesh=mesh, tex=tex.darray[0],
                             caption='with curvature',
                             cblabel='curvature', visb_sc=visb_sc)
visb_sc = splt.visbrain_plot(mesh=mesh, tex=tex.darray[0],
                             caption='change cmap', cblabel='curvature',
                             cmap='hot', visb_sc=visb_sc)
visb_sc.preview()
# then save the 3D rendering figure
# visb_sc.screenshot('test.png')
# # most simple mesh visualization
# splt.pyglet_plot(mesh)
# # with a texture
# splt.pyglet_plot(mesh, tex.darray[0], plot_colormap=True)
# # change in colormap
# splt.pyglet_plot(mesh, tex.darray[0], color_map=plt.get_cmap('hot', 6),
#                  plot_colormap=True)
# # to save to disc as png, we need to get the output of the plot function
# plot_output = splt.pyglet_plot(mesh, tex.darray[0],
#                                color_map=plt.get_cmap('hot'),
#                                plot_colormap=True)
# # then save the 3D rendering figure
# # splt.save_image(plot_output[0], png_fig_filename)
# # and eventually the colobar
# # plot_output[1].savefig(colormap_png_filename)
#
# # trimesh rendering possibilities
# """ set each facet to a random color
# colors are 8 bit RGBA by default (n,4) np.uint8
# for facet in mesh.facets:
#     mesh.visual.face_colors[facet] = trimesh.visual.random_color()
# """
# for vert_ind in range(len(mesh.visual.vertex_colors)):
#     mesh.visual.vertex_colors[vert_ind] = trimesh.visual.random_color()
#
# # default visualization from trimesh
# mesh.show()
