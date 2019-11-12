import trimesh
import slam.plot as splt
import slam.io as sio
import time
import slam.curvature as scurv

if __name__ == '__main__':
    "Ellipse example"
    """
    mesh = sio.load_mesh('ellipsoide.gii')
    "print(mesh.vertices)"
    "print(mesh.faces)"

    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = GetCurvaturesAndDerivatives(mesh)
    print("PrincipalCurvatures", np.shape(PrincipalCurvatures))
    print("Principale dir 1", np.shape(PrincipalDir1))
    print("Principale dir 2", np.shape(PrincipalDir2))
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    # {print("gaussian_curv", gaussian_curv)
    # print("mean_curv", mean_curv)
    # print("mesh.vertex_normals", mesh.vertex_normals)

    M = np.array([[1., 2., 3., 4.], [1., 2., 3., 4.]])
    x = np.transpose(np.sqrt(1 / somme_colonnes(np.transpose(M * M))))
    print(x)
    print(np.shape(x))
    print(np.reshape(x, (2, 1)))
    print(normr(M))
    mesh.show()
    pyglet_plot(mesh, np.transpose(gaussian_curv))
    pyglet_plot(mesh, mean_curv)

    "Sphere example"
    mesh = sio.load_mesh('sphere.gii')
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = GetCurvaturesAndDerivatives(mesh)
    print("PrincipalCurvatures", np.shape(PrincipalCurvatures))
    print("Principale dir 1", np.shape(PrincipalDir1))
    print("Principale dir 2", np.shape(PrincipalDir2))
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    mesh.show()
    pyglet_plot(mesh, np.transpose(gaussian_curv))
    pyglet_plot(mesh, mean_curv)

    "Quad  100 example"
    mesh = sio.load_mesh('quadric_K1_1_K2_-1_10.gii')
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = GetCurvaturesAndDerivatives(mesh)
    print("PrincipalCurvatures", np.shape(PrincipalCurvatures))
    print("Principale dir 1", np.shape(PrincipalDir1))
    print("Principale dir 2", np.shape(PrincipalDir2))
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    mesh.show()
    pyglet_plot(mesh, np.transpose(gaussian_curv))
    pyglet_plot(mesh, mean_curv)

    "Quad  2500 example"
    mesh = sio.load_mesh('quadric_K1_0_K2_1.gii')
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = GetCurvaturesAndDerivatives(mesh)
    print("PrincipalCurvatures", np.shape(PrincipalCurvatures))
    print("Principale dir 1", np.shape(PrincipalDir1))
    print("Principale dir 2", np.shape(PrincipalDir2))
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    mesh.show()
    pyglet_plot(mesh, np.transpose(gaussian_curv))
    pyglet_plot(mesh, mean_curv)

    "Quad  225000 example"
    mesh = sio.load_mesh('quadric_K1_-1_K2_-1_150.gii')
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = GetCurvaturesAndDerivatives(mesh)
    print("PrincipalCurvatures", np.shape(PrincipalCurvatures))
    print("Principale dir 1", np.shape(PrincipalDir1))
    print("Principale dir 2", np.shape(PrincipalDir2))
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    mesh.show()
    pyglet_plot(mesh, np.transpose(gaussian_curv))
    pyglet_plot(mesh, mean_curv)

    white_left_327680 sample
    mesh = sio.load_mesh('OAS1_0006_Lhemi.gii')
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = GetCurvaturesAndDerivatives(mesh)
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    print(gaussian_curv)
    print(mean_curv)
    mesh.show()
    print("min cmean", min(mean_curv))
    print("max cmean", max(mean_curv))
    print("min cgauss", min(gaussian_curv))
    print("max cgauss", max(gaussian_curv))
    pyglet_plot(mesh, gaussian_curv)
    pyglet_plot(mesh, mean_curv)
"""
    t1 = time.time()
    #mesh = sio.load_mesh('hex_quad_k1_-1_k2_-1_10.gii')
    mesh_file = 'data/example_mesh.gii'
    mesh = sio.load_mesh(mesh_file)
    # fd = '/hpc/meca/users/bohi.a/Data/Models/week23_ref_surf/B0.txt'
    # coords = np.loadtxt(fd,skiprows=1,max_rows=50943)
    # faces = np.loadtxt(fd, skiprows=50945, dtype=np.int)
    print(mesh.vertices.shape)
    # mesh = trimesh.Trimesh(faces=faces-1, vertices=coords, process=False)
    mesh.show()
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.GetCurvaturesAndDerivatives(
        mesh)
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    # tex2 = texture.TextureND(darray=mean_curv)
    # sio.write_texture(tex2,'mean_curv.gii')
    print("le calcul a duré : ", time.time() - t1)
    print(gaussian_curv)
    print(mean_curv)
    splt.pyglet_plot(mesh, mean_curv, plot_colormap=True)

    print("min cmean", min(mean_curv))
    print("max cmean", max(mean_curv))
    print("min cgauss", min(gaussian_curv))
    print("max cgauss", max(gaussian_curv))
# pyglet_plot(mesh, gaussian_curv)
# pyglet_plot(mesh, mean_curv)
