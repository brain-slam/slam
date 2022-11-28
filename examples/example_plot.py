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
# The visu is based on visbrain: https://github.com/EtienneCmb/visbrain

###############################################################################
# importation of slam modules
import slam.plot as splt
import slam.io as sio

###############################################################################
# loading an examplar mesh and corresponding texture
mesh_file = "../examples/data/example_mesh.gii"
texture_file = "../examples/data/example_texture.gii"
mesh = sio.load_mesh(mesh_file)
tex = sio.load_texture(texture_file)

###############################################################################
# here is the range of values in the texture:
print("[{a:2.3f}, {b:2.3f}]".format(a=tex.min(), b=tex.max()))

###############################################################################
# plot only the mesh geometry
visb_sc = splt.visbrain_plot(mesh=mesh, caption="simple mesh")
visb_sc.preview()

###############################################################################
# plot the mesh with the curvature as a texture
visb_sc = splt.visbrain_plot(
    mesh=mesh, tex=tex.darray[0], caption="with curvature", cblabel="curvature"
)
visb_sc.preview()

###############################################################################
# change the colormap
visb_sc2 = splt.visbrain_plot(
    mesh=mesh, tex=tex.darray[0], caption="change cmap", cblabel="curvature", cmap="hot"
)
visb_sc2.preview()

###############################################################################
# combine two plots in one single figure, allowing for sinchronization
visb_sc = splt.visbrain_plot(
    mesh=mesh, tex=tex.darray[0], caption="with curvature", cblabel="curvature"
)
visb_sc = splt.visbrain_plot(
    mesh=mesh,
    tex=tex.darray[0],
    caption="change cmap",
    cblabel="curvature",
    cmap="hot",
    visb_sc=visb_sc,
)
visb_sc.preview()

# save the 3D rendering figure
# visb_sc.screenshot('test.png')

###############################################################################
# another option for plotting in slam is using pyglet from Trimseh
# # splt.pyglet_plot(mesh)
# # with a texture
# # splt.pyglet_plot(mesh, tex.darray[0], plot_colormap=True)
# # change in colormap
# # splt.pyglet_plot(mesh, tex.darray[0], color_map=plt.get_cmap('hot', 6),
# #                 plot_colormap=True)
# # to save to disc as png, we need to get the output of the plot function
# # plot_output = splt.pyglet_plot(mesh, tex.darray[0],
# #                                color_map=plt.get_cmap('hot'),
# #                                plot_colormap=True)
# # then save the 3D rendering figure
# # splt.save_image(plot_output[0], png_fig_filename)
# # and eventually the colobar
# # plot_output[1].savefig(colormap_png_filename)
#
# # trimesh rendering possibilities
# # """ set each facet to a random color
# # colors are 8 bit RGBA by default (n,4) np.uint8
# # for facet in mesh.facets:
# #     mesh.visual.face_colors[facet] = trimesh.visual.random_color()
# # """
# # for vert_ind in range(len(mesh.visual.vertex_colors)):
# #     mesh.visual.vertex_colors[vert_ind] = trimesh.visual.random_color()
#
# # default visualization from trimesh
# # mesh.show()
