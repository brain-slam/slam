import trimesh
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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
                plot_colormap=False):
    """
    Visualize a trimesh object using pyglet as proposed in trimesh
    the added value is for texture visualization
    :param mesh: trimesh object
    :param values: numpy array of a texture to be visualized on the mesh
    :param color_map: str, matplotlib colormap, default is 'jet'
    :param plot_colormap: Boolean, if True use matplotlib to plot the colorbar
     of the map on a separate figure
    :return:
    """

    if values is not None:
        if color_map is None:
            color_map = 'jet'

        vect_col_map = \
            trimesh.visual.color.interpolate(values, color_map=color_map)

        if values.shape[0] == mesh.vertices.shape[0]:
            mesh.visual.vertex_colors = vect_col_map
        elif values.shape[0] == mesh.faces.shape[0]:
            mesh.visual.face_colors = vect_col_map

        if plot_colormap:
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(1, 1)
            # fig.subplots_adjust(top=0.95, bottom=0.05, left=0.01, right=0.99)

            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(color_map))
            pos = list(ax.get_position().bounds)
            y_text = pos[1] + pos[3] / 2.
            fig.text(pos[0] - 0.01, y_text, '{:0.2f}'.format(np.min(values)),
                     va='center', ha='right', fontsize=15, color='k')
            fig.text(pos[2] + pos[0] + 0.01, y_text,
                     '{:0.2f}'.format(np.max(values)),
                     va='center', fontsize=15, color='k')
            ax.set_axis_off()
            plt.show()
    # call the default trimesh visualization tool using pyglet

    mesh.show(background=[0, 0, 0, 255])
