import slam.plot as splt
import slam.io as sio

if __name__ == '__main__':

    mesh_file = 'data/example_mesh.gii'
    texture_file = 'data/example_texture.gii'

    mesh = sio.load_mesh(mesh_file)

    mesh.apply_transform(mesh.principal_inertia_transform)
    tex = sio.load_texture(texture_file)
    print(tex.min())
    print(tex.max())
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
