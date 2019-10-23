import slam.io as sio
import slam.differential_geometry as sdg

if __name__ == '__main__':
    mesh = sio.load_mesh('data/example_mesh.gii')

    lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='fem')
    print(mesh.vertices.shape)
    print(lap.shape)
    # import slam.plot as splt
    # splt.pyglet_plot(mesh, lap[:, 0].todense().squeeze())
