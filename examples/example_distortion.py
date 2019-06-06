import slam.distortion as sdst
from trimesh import smoothing as sm
import slam.plot as splt
import slam.io as sio
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')

if __name__ == '__main__':

    mesh = sio.load_mesh('data/example_mesh.gii')
    mesh.apply_transform(mesh.principal_inertia_transform)
    # mesh.show()
    mesh_s = sm.filter_laplacian(mesh.copy(), iterations=100)
    # mesh_s.show()

    print(mesh.vertices.shape)

    angle_diff = sdst.angle_difference(mesh, mesh_s)
    print(angle_diff)

    face_angle_dist = np.sum(angle_diff, 1)
    print(face_angle_dist)

    vect_col = np.random.random_integers(0, 255, mesh.faces.shape[0])
    print(vect_col.shape)
    # mesh.visual.vertex_colors = vert_col
    mesh.visual.faces_colors = vect_col
    mesh.show()

    # f, ax = plt.subplots(1,1)
    # ax.set_title('angles')
    # ax.hist(angle_diff.flatten())
    # # axs[3].set_xticks(X_edge)
    # ax.grid(True)
    # plt.show()

    splt.pyglet_plot(mesh_s, face_angle_dist, 'hot', True)
