import trimesh
import slam.plot as splt
import slam.io as sio
import numpy as np

if __name__ == '__main__':

    mesh_1 = sio.load_mesh('data/example_mesh.gii')
    mesh_2 = sio.load_mesh('data/example_mesh_2.gii')

    print('computing ICP registration')
    transf_mat, cost = trimesh.registration.mesh_other(mesh_1, mesh_2,
                                                       samples=500,
                                                       scale=False,
                                                       icp_first=10,
                                                       icp_final=100)
    print(transf_mat)
    print(cost)
    # make the figures
    joint_mesh = mesh_1 + mesh_2
    joint_tex = np.ones((joint_mesh.vertices.shape[0],))
    joint_tex[:mesh_1.vertices.shape[0]] = 10
    visb_sc = splt.visbrain_plot(mesh=joint_mesh, tex=joint_tex,
                                 caption='before registration')
    mesh_1.apply_transform(transf_mat)
    joint_mesh = mesh_1 + mesh_2
    visb_sc = splt.visbrain_plot(mesh=joint_mesh, tex=joint_tex,
                                 caption='after registration', visb_sc=visb_sc)
    visb_sc.preview()
