import trimesh
import numpy as np


def save_image(scene_viewer, filename):
    """
    save a slam figure to disc
    :param scene:
    :param filename:
    :return:
    """
    scene = scene_viewer._scene
    with open(filename, 'wb') as f:
        f.write(scene.save_image(background=scene_viewer.background))


def visbrain_plot(mesh, tex=None, caption=None, cblabel=None, visb_sc=None,
                  cmap='jet'):
    """
    Visualize a trimesh object using visbrain core plotting tool
    :param mesh: trimesh object
    :param tex: numpy array of a texture to be visualized on the mesh
    :return:
    """
    from visbrain.objects import BrainObj, ColorbarObj, SceneObj
    b_obj = BrainObj('gui', vertices=np.array(mesh.vertices),
                     faces=np.array(mesh.faces),
                     translucent=False)
    if not isinstance(visb_sc, SceneObj):
        visb_sc = SceneObj(bgcolor='black', size=(1000, 1000))
    # identify (row, col)
    row, _ = get_visb_sc_shape(visb_sc)
    visb_sc.add_to_subplot(b_obj, row=row, col=0, title=caption)

    if tex is not None:
        b_obj.add_activation(data=tex, cmap=cmap,
                             clim=(np.min(tex), np.max(tex)))
        CBAR_STATE = dict(cbtxtsz=20, txtsz=20., width=.1, cbtxtsh=3.,
                          rect=(-.3, -2., 1., 4.), cblabel=cblabel)
        cbar = ColorbarObj(b_obj, **CBAR_STATE)
        visb_sc.add_to_subplot(cbar, row=row, col=1, width_max=200)

    return visb_sc


def get_visb_sc_shape(visb_sc):
    """
    get the subplot shape in a visbrain scene
    :param visb_sc:
    :return: tuple (number of rows, number of columns)
    """
    vb_shape = visb_sc._grid_desc.keys()
    if not len(vb_shape):
        rc = (0, 0)
    else:
        rc = (max([k[0] for k in vb_shape]), max([k[1] for k in vb_shape]))
    return rc


def pyglet_plot(mesh_in, values=None, color_map=None,
                plot_colormap=False, caption=None,
                alpha_transp=255, background_color=None,
                default_color=[100, 100, 100, 200], cmap_bounds=None):
    """
    Visualize a trimesh object using pyglet as proposed in trimesh
    the added value is for texture visualization
    :param mesh_in: trimesh object
    :param values: numpy array of a texture to be visualized on the mesh
    :param color_map: str, matplotlib colormap, default is 'jet'
    :param plot_colormap: Boolean, if True use matplotlib to plot the colorbar
     of the map on a separate figure
    :param caption: Title of window
    :param alpha_transp: mesh transparency parameter, 0=transparent, 255=solid
    :param background_color:
    :param default_color: color of vertices or faces where 'values' is NaN
    :param cmap_bounds : bounds to impose on the colormap
    :return:
    """
    import matplotlib.pyplot as plt
    # to ensure plotting do not affect the mesh (esp. visual aspects)
    mesh = mesh_in.copy()
    if background_color is not None:
        background = background_color
    else:
        # default background color is black
        background = [0, 0, 0, 255]
        # turn black the background of colormap fig
        plt.style.use('dark_background')
    fig = None
    smooth = True
    if values is None:
        mesh.visual.vertex_colors = default_color
    else:
        if color_map is None:
            color_map = plt.get_cmap('jet', 12)
        # in case NaN are present in 'values'
        nan_inds = np.isnan(values)
        if sum(nan_inds) == len(values):
            print('no value in the texture')
            vect_col_map = np.tile(default_color, (1, len(values)))
        else:

            vect_col_map_tmp = \
                trimesh.visual.color.interpolate(values[~nan_inds],
                                                 color_map=color_map)
            vect_col_map = np.zeros((len(values), 4))
            vect_col_map[~nan_inds, :] = vect_col_map_tmp
            vect_col_map[:, 3] = alpha_transp
            vect_col_map[nan_inds, :] = default_color

        if values.shape[0] == mesh.vertices.shape[0]:
            mesh.visual.vertex_colors = vect_col_map
        elif values.shape[0] == mesh.faces.shape[0]:
            mesh.visual.face_colors = vect_col_map
            smooth = False

        if plot_colormap:
            import matplotlib as mpl
            mpl.rcParams.update({'font.size': 20})
            # fig = plt.figure(figsize=(8, 2))
            # ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
            fig, ax = plt.subplots(1, 1)
            ax.set_title(caption)

            if cmap_bounds is None:
                # vmin = np.around(np.min(values[~nan_inds]))
                # vmax = np.around(np.max(values[~nan_inds]))
                vmin = np.min(values[~nan_inds])
                vmax = np.max(values[~nan_inds])
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = mpl.colors.BoundaryNorm(cmap_bounds, color_map.N)
            # print(norm)
            mpl.colorbar.ColorbarBase(ax, cmap=color_map,norm=norm,
                                      orientation='vertical') # L'affiche merdouille, que des valeurs dans [0,1]

            # mpl.colorbar.ColorbarBase(ax, cmap=color_map,
            #                           orientation='horizontal')
            fig.set_size_inches(18, 4)
            plt.show()

            # gradient = np.linspace(0, 1, 256)
            # gradient = np.vstack((gradient, gradient))
            # fig, ax = plt.subplots(1, 1)
            # # fig.subplots_adjust(top=0.95, bottom=0.05, left=0.01,
            # right=0.99)
            #
            # ax.set_title(caption)
            # ax.imshow(gradient, aspect='auto', cmap=color_map)
            # pos = list(ax.get_position().bounds)
            # y_text = pos[1] + pos[3] / 2.
            # fig.text(pos[0] - 0.01, y_text,
            #          '{:0.0000009f}'.format(np.min(values[~nan_inds])),
            #          va='center', ha='right', fontsize=15, color='k')
            # fig.text(pos[2] + pos[0] + 0.01, y_text,
            #          '{:0.0000009f}'.format(np.max(values[~nan_inds])),
            #          va='center', fontsize=15, color='k')
            # ax.set_axis_off()
            # fig.set_size_inches(18, 3)
            # plt.show()

    # call the default trimesh visualization tool using pyglet
    # light = trimesh.scene.lighting.DirectionalLight()
    # [trimesh.scene.lighting.Light]

    scene = trimesh.Scene(mesh)  # , lights=[light])
    # print(scene.graph)
    # #lights, transforms = lighting.autolight(self)
    # lights, transforms = trimesh.scene.lighting.autolight(scene)
    # # assign the transforms to the scene graph
    # for L, T in zip(lights, transforms):
    #     L.intensity = 10000
    #     scene.graph[L.name] = T
    #     # set the lights
    # scene._lights = lights
    scene_viewer = scene.show(caption=caption,
                              smooth=smooth,
                              background=background)
    if fig is None:
        output = [scene_viewer]
    else:
        output = [scene_viewer, fig]
    return output
