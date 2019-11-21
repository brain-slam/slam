import slam.distortion as sdst
import slam.differential_geometry as sdg
import slam.plot as splt
import slam.io as sio
import numpy as np

if __name__ == '__main__':

    mesh = sio.load_mesh('data/example_mesh.gii')
    mesh_s = sdg.laplacian_mesh_smoothing(mesh, nb_iter=50, dt=0.1)

    print(mesh.vertices.shape)

    angle_diff = sdst.angle_difference(mesh, mesh_s)
    print(angle_diff)

    face_angle_dist = np.sum(np.abs(angle_diff), 1)
    print(face_angle_dist)

    vect_col = np.random.random_integers(0, 255, mesh.faces.shape[0])
    print(vect_col.shape)

    # f, ax = plt.subplots(1,1)
    # ax.set_title('angles')
    # ax.hist(angle_diff.flatten())
    # # axs[3].set_xticks(X_edge)
    # ax.grid(True)
    # plt.show()

    # splt.pyglet_plot(mesh_s, face_angle_dist, 'hot', True)

    visb_sc = splt.visbrain_plot(mesh=mesh, caption='original mesh')
    visb_sc = splt.visbrain_plot(mesh=mesh_s, caption='smoothed mesh',
                                 visb_sc=visb_sc)
    # TODO plot distortion map onto the mesh
    # visb_sc = splt.visbrain_plot(mesh=mesh_s, tex=face_angle_dist,
    # caption='face angle distortions', cblabel='angle distortions',
    # cmap='hot', visb_sc=visb_sc)
    visb_sc.preview()
