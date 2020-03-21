import slam.generate_parametric_surfaces as sps
import numpy as np
import slam.topology as stop
import slam.plot as splt
import slam.mapping as smap
import slam.distortion as sdst
from vispy.scene import Line
from visbrain.objects import VispyObj, SourceObj

if __name__ == '__main__':
    # let us generate an open mesh
    K = [-1, -1]
    open_mesh = sps.generate_quadric(K, nstep=5)
    open_mesh.show()
    open_mesh_boundary = stop.mesh_boundary(open_mesh)

    # mapping onto a planar disk
    disk_mesh = smap.disk_conformal_mapping(open_mesh)
    disk_mesh.show()
    # compute distortion measures between original and planar representations
    angle_diff = sdst.angle_difference(disk_mesh, open_mesh)
    area_diff = sdst.area_difference(disk_mesh, open_mesh)
    edge_diff = sdst.edge_length_difference(disk_mesh, open_mesh)
    print(np.mean(angle_diff))
    print(np.mean(area_diff))
    print(np.mean(edge_diff))

    visb_sc = splt.visbrain_plot(mesh=open_mesh, caption='open mesh')
    # create points with vispy
    for bound in open_mesh_boundary:
        points = open_mesh.vertices[bound]
        s_rad = SourceObj('rad', points, color='red', symbol='square',
                          radius_min=10)
        visb_sc.add_to_subplot(s_rad)
        lines = Line(pos=open_mesh.vertices[bound], width=10, color='b')
        # wrap the vispy object using visbrain
        l_obj = VispyObj('line', lines)
        visb_sc.add_to_subplot(l_obj)

    visb_sc2 = splt.visbrain_plot(mesh=disk_mesh, caption='disk mesh')
    # create points with vispy
    for bound in open_mesh_boundary:
        points = disk_mesh.vertices[bound]
        s_rad = SourceObj('rad', points, color='red', symbol='square',
                          radius_min=10)
        visb_sc2.add_to_subplot(s_rad)
        lines = Line(pos=disk_mesh.vertices[bound], width=10, color='b')
        # wrap the vispy object using visbrain
        l_obj = VispyObj('line', lines)
        visb_sc2.add_to_subplot(l_obj)
    visb_sc.preview()
    visb_sc2.preview()
    #
    # mesh = sio.load_mesh('data/example_mesh.gii')
    # sphere, evol = smap.spherical_mapping(mesh, mapping_type='conformal',
    #                                       dt=0.01, nb_it=3000)
    # visb_sc = splt.visbrain_plot(mesh=mesh, caption='original mesh')
    # visb_sc = splt.visbrain_plot(mesh=sphere,
    #                              caption='spherical representation',
    #                              visb_sc=visb_sc)
    # visb_sc.preview()
    #
    # angle_diff = sdst.angle_difference(sphere, mesh)
    # area_diff = sdst.area_difference(sphere, mesh)
    # edge_diff = sdst.edge_length_difference(sphere, mesh)
    #
    # aevol = np.array(evol)
    # f, ax = plt.subplots(1, 3)
    # ax[0].set_title('angles')
    # # ax[0].hist(angle_diff.flatten())
    # ax[0].plot(aevol[:, 0])
    # ax[0].grid(True)
    # ax[1].set_title('areas')
    # # ax[1].hist(area_diff.flatten())
    # ax[1].plot(aevol[:, 1])
    # ax[1].grid(True)
    # ax[2].set_title('edges')
    # # ax[2].hist(edge_diff.flatten())
    # ax[2].plot(aevol[:, 2])
    # ax[2].grid(True)
    # #
    # # sph, evol = smap.spherical_mapping(mesh, mapping_type='authalic',
    # # dt=0.01, nb_it=1000)
    # #
    # # angle_diff = sdst.angle_difference(sph, mesh)
    # # area_diff = sdst.area_difference(sph, mesh)
    # # edge_diff = sdst.edge_length_difference(sph, mesh)
    # #
    # # f, ax = plt.subplots(1, 3)
    # # ax[0].set_title('angles')
    # # ax[0].hist(angle_diff.flatten())
    # # ax[0].grid(True)
    # # ax[1].set_title('areas')
    # # ax[1].hist(area_diff.flatten())
    # # ax[1].grid(True)
    # # ax[2].set_title('edges')
    # # ax[2].hist(edge_diff.flatten())
    # # ax[2].grid(True)
    # #
    # # sph = smap.spherical_mapping(mesh)
    # #
    # # angle_diff = sdst.angle_difference(sph, mesh)
    # # area_diff = sdst.area_difference(sph, mesh)
    # # edge_diff = sdst.edge_length_difference(sph, mesh)
    # #
    # # f, ax = plt.subplots(1, 3)
    # # ax[0].set_title('angles')
    # # ax[0].hist(angle_diff.flatten())
    # # ax[0].grid(True)
    # # ax[1].set_title('areas')
    # # ax[1].hist(area_diff.flatten())
    # # ax[1].grid(True)
    # # ax[2].set_title('edges')
    # # ax[2].hist(edge_diff.flatten())
    # # ax[2].grid(True)
    # plt.show()
    #
    # sphere_mesh = sps.generate_sphere(1000)
    # print(np.mean(sphere_mesh.vertices))
    # print(np.sqrt(np.sum(np.power(sphere_mesh.vertices, 2), 1)))
    # z_coord_texture = np.array(sphere_mesh.vertices[:, 2])
    #
    # poly_angles = meshPolygonAngles(sphere_mesh.vertices, sphere_mesh.faces)
    # print(np.max(poly_angles))
    # print(np.min(poly_angles))
    #
    # visb_sc_2 = splt.visbrain_plot(mesh=sphere_mesh, tex=z_coord_texture,
    #                                caption='parametric sphere',
    #                                cblabel='z coordinate')
    # visb_sc_2.preview()

    #
    # # plane_proj_mesh = sphmap.stereo_projection(sphere_mesh, invert=False)
    # # splt.pyglet_plot(plane_proj_mesh,
    # # z_coord_texture, caption="projected onto a plane")
    #
    # # plane_proj_mesh = sphmap.stereo_projection(sphere_mesh)
    # # splt.pyglet_plot(plane_proj_mesh, z_coord_texture,
    # #                  caption="projected onto a plane")
    # #
    # # inv_plane_proj_mesh = sphmap.inverse_stereo_projection(plane_proj_mesh)
    # # splt.pyglet_plot(inv_plane_proj_mesh, z_coord_texture,
    # #                  caption="inverse projected onto a plane")
    #
    # b = complex(0., 0.)
    # c = complex(0., 0.)
    # d = complex(1., 0.)
    # t = -1
    # step = +0.1
    # all_steps = list()
    # all_spheres = list()
    # all_angle_diff = list()
    # all_area_diff = list()
    # plane_proj_mesh = smap.stereo_projection(sphere_mesh, invert=False)
    # for i in range(8):
    #     t += step
    #     all_steps.append(t)
    #     a = complex(t, 0)
    #
    #     plan_complex_transfo = smap.moebius_transformation(a, b, c, d,
    #                                                        plane_proj_mesh)
    #     sphere_transformed_mesh = smap.inverse_stereo_projection(
    #         plan_complex_transfo, invert=False)
    #     all_spheres.append(sphere_transformed_mesh)
    #     angle_diff = sdst.angle_difference(sphere_mesh,
    #                                        sphere_transformed_mesh)
    #     all_angle_diff.append(angle_diff)
    #     area_diff = sdst.area_difference(sphere_mesh,
    #                                      sphere_transformed_mesh)
    #     all_area_diff.append(area_diff)
    #     poly_angles = meshPolygonAngles(sphere_transformed_mesh.vertices,
    #                                     sphere_transformed_mesh.faces)
    #     print(np.max(sphere_transformed_mesh.face_angles - poly_angles))
    #     poly_areas = meshPolygonArea(sphere_transformed_mesh.vertices,
    #                                  sphere_transformed_mesh.faces)
    #     print(np.max(sphere_transformed_mesh.area_faces - poly_areas))
    # splt.pyglet_plot(sphere_transformed_mesh, z_coord_texture,
    #                  caption="moebius transformed sphere")
    #
    # int_area = [np.sum(np.abs(v)) for v in all_area_diff]
    # int_angle = [np.sum(np.abs(angle_diff).flatten()) for v in
    #              all_angle_diff]
    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('moebius scaling parameter')
    # funct1 = ax.plot(all_steps, int_area, 'o-', label='area')
    # ax.set_ylabel('integral of area distortionss')
    #
    # # Share the x-axis for both the axes (ax1, ax2)
    # ax2 = ax.twinx()
    # funct2 = ax2.plot(all_steps, int_angle, 'rs-', label='angle')
    # ax2.set_ylabel('integral of angle distortions')
    # ax2.set_ylim([-0.1, 3.14])
    #
    # functs = funct1 + funct2
    # labels = [f.get_label() for f in functs]
    # plt.legend(functs, labels)
    # ax.grid()
    #
    # plt.show()
