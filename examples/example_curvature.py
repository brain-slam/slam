import trimesh
import slam.plot as splt
import slam.curvature as get_curvatures

if __name__ == '__main__':
    # Generate a sphere
    mesh = trimesh.creation.icosphere()

    # Show th sphere
    mesh.show()

    # Calculate Rusinkiewicz estimation of mean and gauss curvatures
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
        get_curvatures.getcurvaturesandderivatives(mesh)
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

    # Plot mean curvature
    splt.pyglet_plot(mesh, mean_curv, plot_colormap=True)
    # PLot Gauss curvature
    splt.pyglet_plot(mesh, gaussian_curv, plot_colormap=True)
