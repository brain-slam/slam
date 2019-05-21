import numpy as np


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


def linear_interp_rgba_colormap(val_color_a, val_color_b, res=256):
    """
    linear interpolation to create colormaps
    :param val_color_a: list of length 5 setting the value and corresponding
    RGBA color for the inferior bound ot he colormap.
    :param val_color_b: list of length 5 setting the value and corresponding
    RGBA color for the superior bound ot he colormap.
    :param res: number of intervals in the returned colormap
    :return: the resulting colormap in the form of a value-RGBAcolor matrix of
    size res*5 (formatted as list of list)
    """
    acolor_a = np.array(val_color_a[1:])
    acolor_b = np.array(val_color_b[1:])
    val_a = val_color_a[0]
    val_b = val_color_b[0]
    vals = [-np.Inf]
    for t in range(res - 1):
        vals.append(val_a + (val_b - val_a) * t / (res - 2))

    colors = list()
    for t in range(res):
        colors.append(np.round(acolor_a + (acolor_b - acolor_a)
                               * t / (res - 1)).tolist())
    val_colors = []
    for c, v in zip(colors, vals):
        val_colors.append([v] + c)
    return val_colors


def pyglet_plot(mesh, map=None, map_min=None, map_max=None, plot_colormap=False):
    """
    Visualize a trimesh object using pyglet as proposed in trimesh
    the added value is for texture visualization
    :param mesh: trimesh object
    :param map: numpy array of a texture to be visualized on the mesh
    :return:
    """

    if map is not None:
        # scale the map between 0 and 1
        scaled_curv = map - map.min()
        scaled_curv = scaled_curv / scaled_curv.max()
        # convert into uint8 in [0 255]
        # vect_col = np.stack([255 * np.ones(scaled_curv.shape),
        #                      np.round(scaled_curv * 255),
        #                      np.round(scaled_curv * 255),
        #                      255 * np.ones(scaled_curv.shape)],
        #                     axis=1).astype(np.uint8)
        mean_map_val = 0
        if map_max is None:
            max_map_val = np.max(np.abs(map))
        else:
            max_map_val = map_max
            mean_map_val = map.mean()
        if map_min is None:
            min_map_val = -max_map_val
        else:
            min_map_val = map_min

        clmap_neg = linear_interp_rgba_colormap(
            [min_map_val, 0, 0, 255, 255],
            [mean_map_val, 255, 255, 255, 255], res=128)
        clmap_pos = linear_interp_rgba_colormap(
            [mean_map_val, 255, 255, 255, 255],
            [max_map_val, 255, 0, 0, 255], res=128)
        clmap = clmap_neg
        clmap.extend(clmap_pos[1:])

        vect_col_map = list()
        for val in map:
            for c in clmap:
                if val > c[0]:  # < c[0]:
                    color = c[1:]
            vect_col_map.append(color)
        vect_col_map = np.array(vect_col_map, dtype=np.uint8)
        if map.shape[0] == mesh.vertices.shape[0]:
            # vect_col  # color.to_rgba(vect_col)
            mesh.visual.vertex_colors = vect_col_map
        elif map.shape[0] == mesh.faces.shape[0]:
            mesh.visual.face_colors = vect_col_map

        if plot_colormap:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            a_clmap = np.array(clmap)
            a_clmap = np.array(a_clmap[:, 1:-1])
            a_clmap = np.array(a_clmap, dtype=np.uint8)
            img = np.tile(a_clmap, (20,1,1))
            plt.imshow(img)
            plt.text(5, 12, '{:0.2f}'.format(min_map_val), color='w')
            plt.text(230, 12, '{:0.2f}'.format(max_map_val), color='w')
            plt.axis('off')
            plt.show()
    # call the default trimesh visualization tool using pyglet

    mesh.show()
