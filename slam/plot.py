import trimesh
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def save_image(scene, filename):
    """
    save a slam figure to disc
    :param scene:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as f:
        f.write(scene.save_image())


def visbrain_plot(mesh, tex=None):
    """
    Visualize a trimesh object using visbrain core plotting tool
    :param mesh: trimesh object
    :param tex: numpy array of a texture to be visualized on the mesh
    :return:
    """
    from visbrain.objects import BrainObj

    # invert_normals = True -> Light outside
    # invert_normals = False -> Light inside
    b_obj = BrainObj('gui', vertices=mesh.vertices, faces=mesh.faces,
                     translucent=False, invert_normals=True)
    if tex is not None:
        b_obj.add_activation(data=tex, cmap='viridis')
    b_obj.preview(bgcolor='white')


def pyglet_plot(mesh, values=None, color_map=None,
                plot_colormap=False, caption=None,
                alpha_transp=255, background_color=None):
    """
    Visualize a trimesh object using pyglet as proposed in trimesh
    the added value is for texture visualization
    :param mesh: trimesh object
    :param values: numpy array of a texture to be visualized on the mesh
    :param color_map: str, matplotlib colormap, default is 'jet'
    :param plot_colormap: Boolean, if True use matplotlib to plot the colorbar
     of the map on a separate figure
    :param caption: Title of window
    :param alpha_transp: mesh transparency parameter, 0=fully transparent, 255=solid
    :param background_color:
    :return:
    """
    if background_color is not None:
        background = background_color
    else:
        background = [0, 0, 0, 255]

    if values is not None:
        smooth = True
        if color_map is None:
            color_map = 'jet'

        vect_col_map = \
            trimesh.visual.color.interpolate(values, color_map=color_map)
        vect_col_map[:, 3] = alpha_transp
        if values.shape[0] == mesh.vertices.shape[0]:
            mesh.visual.vertex_colors = vect_col_map
        elif values.shape[0] == mesh.faces.shape[0]:
            mesh.visual.face_colors = vect_col_map
            smooth = False

        if plot_colormap:
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(1, 1)
            # fig.subplots_adjust(top=0.95, bottom=0.05, left=0.01, right=0.99)

            ax.set_title(caption)
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(color_map))
            pos = list(ax.get_position().bounds)
            y_text = pos[1] + pos[3] / 2.
            fig.text(pos[0] - 0.01, y_text,
                     '{:0.0000009f}'.format(np.min(values)),
                     va='center', ha='right', fontsize=15, color='k')
            fig.text(pos[2] + pos[0] + 0.01, y_text,
                     '{:0.0000009f}'.format(np.max(values)),
                     va='center', fontsize=15, color='k')
            ax.set_axis_off()
            fig.set_size_inches(18, 3)
            plt.show()

    # call the default trimesh visualization tool using pyglet
    scene = trimesh.Scene(mesh)
    scene.show(caption=caption, smooth=smooth, background=background)
    if fig is None:
        output = [scene]
    else:
        output = [scene, fig]
    return output
