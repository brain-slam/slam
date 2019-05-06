import trimesh
import slam.plot as splt
import slam.io as sio

if __name__ == '__main__':
    mesh_file = 'example_mesh.gii'
    texture_file = 'example_texture.gii'

    mesh = sio.load_mesh(mesh_file)
    mesh.apply_transform(mesh.principal_inertia_transform)
    tex = sio.load_texture(texture_file)

    splt.pyglet_plot(mesh, tex.darray)

    """ set each facet to a random color
    colors are 8 bit RGBA by default (n,4) np.uint8
    for facet in mesh.facets:
        mesh.visual.face_colors[facet] = trimesh.visual.random_color()
    """
    for vert_ind in range(len(mesh.visual.vertex_colors)):
        mesh.visual.vertex_colors[vert_ind] = trimesh.visual.random_color()

    # preview mesh in an opengl window if you installed pyglet with pip
    mesh.show()
