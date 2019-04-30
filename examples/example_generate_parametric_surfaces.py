import trimesh
import slam.generate_parametric_surfaces as sps
import numpy as np

if __name__ == '__main__':

    Ks = [[1, 1]]
    X, Y, faces, Zs = sps.generate_quadric(Ks, nstep=10)
    Z =Zs[0]

    coords = np.array([X,Y,Z]).transpose()
    mesh = trimesh.Trimesh(faces=faces, vertices=coords, process=False)
    mesh.show()
    # Ks = [[1, 1], [1, 0], [0, 0], [-1, 1]]
    # X, Y, Tri, Zs = generate_quadric(Ks)
    # for K1 in range(-1,2,1):
    #     for K2 in range(-1,2,1):
    #         Z=quadric(K1,K2)(X,Y)
    #         coord = np.array([X,Y,Z]).transpose()
    #         print(X.shape)
    #         print(coord.shape)
    #         mesh_from_arrays(coord, Tri.triangles, os.path.join(output_folder,'quadric_K1_'+str(K1)+'_K2_'+str(K2)+'.gii'))

    # Vizualisation
    # fig = plt.figure(figsize=(15, 12))
    # cpt = 1
    # for Z, K in zip(Zs, Ks):
    #     ax = fig.add_subplot(2, 2, cpt, projection='3d')
    #     ax.plot_trisurf(X, Y, Z, triangles=Tri.triangles, shade=True)
    #     ax.view_init(25,45)
    #     ax.set_title(str(K[0])+' '+str(K[1]))
    #     cpt=cpt+1


    # Ellipsoid Parameters
    nstep=50
    randomSampling=True
    a=2
    b=1
    ellips = sps.generate_ellipsiod(a, b, nstep, randomSampling)
    ellips.show()

    sphere = sps.generate_sphere(10)
    sphere.show()


    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
    # ax.set_aspect('equal')
    # # Meshing
    # THETA = np.arcsin(coords[:, 2])
    # PHI = np.arctan(coords[:, 0] / coords[:, 1]) + np.pi * (np.sign(coords[:, 1]) + 2) / 2
    # plt.figure()
    # plt.plot(THETA, PHI, '+')
    #
    # Tri = Triangulation(THETA, PHI)
    #
    # # Visualization
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], triangles=Tri.triangles, shade=True)
    # ax.view_init(25, 25)
    # ax.set_aspect('equal')



#     # # Visualization
#     # fig=plt.figure(figsize=(15,12))
#     # ax = fig.gca(projection='3d')
#     # ax.plot_trisurf(X,Y,Z,triangles=Tri.triangles,shade=True)
#     # ax.view_init(25,25)
#     # ax.set_aspect('equal')
#     # plt.show()
#
