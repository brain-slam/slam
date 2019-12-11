import slam.io as sio
import slam.plot as splt
import slam.curvature as scurv

if __name__ == '__main__':
    mesh_file = 'data/example_mesh.gii'
    mesh = sio.load_mesh(mesh_file)
    # VertexNormals, Avertex, Acorner, up, vp =
    # scurv.calcvertex_normals(mesh, mesh.face_normals)

    mesh.apply_transform(mesh.principal_inertia_transform)
    # uncomment if you want to test the code on a sphere
    # import slam.generate_parametric_surfaces as sgps
    # mesh = sgps.generate_sphere()

    # Calculate Rusinkiewicz estimation of mean and gauss curvatures
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
        scurv.curvatures_and_derivatives(mesh)
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

    # Plot mean curvature
    visb_sc = splt.visbrain_plot(mesh=mesh, tex=mean_curv,
                                 caption='mean curvature',
                                 cblabel='mean curvature')
    # PLot Gauss curvature
    visb_sc = splt.visbrain_plot(mesh=mesh, tex=gaussian_curv,
                                 caption='Gaussian curvature',
                                 cblabel='Gaussian curvature',
                                 cmap='hot', visb_sc=visb_sc)
    visb_sc.preview()
