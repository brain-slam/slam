
import matplotlib.pyplot as plt
import slam.generate_parametric_surfaces as sps
import numpy as np
import slam.plot as splt
import slam.spherical_mapping as sphmap
import slam.distortion as sdst

if __name__ == '__main__':
    sphere_mesh = sps.generate_sphere(100)
    print(np.mean(sphere_mesh.vertices))
    print(np.sqrt(np.sum(np.power(sphere_mesh.vertices, 2), 1)))
    z_coord_texture = sphere_mesh.vertices[:, 2]
    splt.pyglet_plot(sphere_mesh, z_coord_texture, caption="Sphere")

    # plane_proj_mesh = sphmap.stereo_projection(sphere_mesh, invert=False)
    # splt.pyglet_plot(plane_proj_mesh,
    # z_coord_texture, caption="projected onto a plane")

    plane_proj_mesh = sphmap.stereo_projection(sphere_mesh)
    splt.pyglet_plot(plane_proj_mesh, z_coord_texture,
                     caption="projected onto a plane")

    inv_plane_proj_mesh = sphmap.inverse_stereo_projection(plane_proj_mesh)
    splt.pyglet_plot(inv_plane_proj_mesh, z_coord_texture,
                     caption="inverse projected onto a plane")

    b = complex(0., 0.)
    c = complex(0., 0.)
    d = complex(1., 0.)
    t = 1
    step = -0.1
    all_steps = list()
    all_spheres = list()
    all_angle_dist = list()
    all_area_dist = list()
    plane_proj_mesh = sphmap.stereo_projection(sphere_mesh, invert=False)
    for i in range(10):
        print(i)
        t += step
        all_steps.append(t)
        a = complex(t, 0)

        plan_complex_transfo = \
            sphmap.mobius_transformation(a, b, c, d, plane_proj_mesh)
        sphere_transformed_mesh = \
            sphmap.inverse_stereo_projection(plan_complex_transfo,
                                             invert=False)
        all_spheres.append(sphere_transformed_mesh)
        angle_diff = sdst.angle_difference(sphere_mesh,
                                           sphere_transformed_mesh)
        all_angle_dist.append(np.sum(np.abs(angle_diff).flatten()))
        area_diff = sdst.area_difference(sphere_mesh, sphere_transformed_mesh)
        all_area_dist.append(np.sum(np.abs(area_diff)))

    print(all_angle_dist)
    print(all_area_dist)
    fig, ax = plt.subplots(1, 1)
    ax.plot(all_steps, all_angle_dist, 'o-')
    ax.plot(all_steps, all_area_dist, 's-')
    ax.legend(['angle', 'area'])
    plt.show()

    splt.pyglet_plot(sphere_transformed_mesh, z_coord_texture,
                     caption="moebius transformed sphere")
