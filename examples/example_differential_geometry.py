import slam.io as sio
import slam.differential_geometry as sdg

if __name__ == '__main__':
    mesh = sio.load_mesh('data/example_mesh.gii')

    lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='fem')
    print(mesh.vertices.shape)
    print(lap.shape)
    lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='conformal')
    lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='meanvalue')
    lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='authalic')

    s_mesh = sdg.laplacian_mesh_smoothing(mesh, nb_iter=10, dt=0.1)
    # import slam.plot as splt
    # splt.pyglet_plot(mesh, lap[:, 0].todense().squeeze())
