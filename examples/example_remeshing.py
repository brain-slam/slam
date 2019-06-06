import slam.plot as splt
import slam.io as sio
import slam.remeshing as srem


if __name__ == '__main__':
    # source object files
    source_mesh_file = 'data/example_mesh.gii'
    source_texture_file = 'data/example_texture.gii'
    source_spherical_mesh_file = 'data/example_mesh_spherical.gii'
    # target object files
    target_mesh_file = 'data/example_mesh_2.gii'
    target_spherical_mesh_file = 'data/example_mesh_2_spherical.gii'

    source_mesh = sio.load_mesh(source_mesh_file)
    source_tex = sio.load_texture(source_texture_file)
    source_spherical_mesh = sio.load_mesh(source_spherical_mesh_file)
    splt.pyglet_plot(source_mesh, source_tex.darray)
    splt.pyglet_plot(source_spherical_mesh, source_tex.darray)

    target_mesh = sio.load_mesh(target_mesh_file)
    target_spherical_mesh = sio.load_mesh(target_spherical_mesh_file)

    interpolated_tex = \
        srem.spherical_interpolation_nearest_neigbhor(source_spherical_mesh,
                                                      target_spherical_mesh,
                                                      source_tex.darray)
    splt.pyglet_plot(target_mesh, interpolated_tex, plot_colormap=True)
