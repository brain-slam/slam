# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import trimesh
import slam.io as sio
import slam.topology as stop
# import slam.plot as splt
import slam.generate_parametric_surfaces as sps
import numpy as np

if __name__ == '__main__':
    # here is how to get the vertices that define the boundary of an open mesh
    Ks = [[1, 1]]
    X, Y, faces, Zs = sps.generate_quadric(Ks, nstep=10)
    Z = Zs[0]

    coords = np.array([X, Y, Z]).transpose()
    open_mesh = trimesh.Trimesh(faces=faces, vertices=coords, process=False)

    open_mesh_boundary = stop.mesh_boundary(open_mesh)
    print(open_mesh_boundary)
    scene_list = [open_mesh]
    for bound in open_mesh_boundary:
        points = open_mesh.vertices[bound]
        cloud_boundary = trimesh.points.PointCloud(points)
        cloud_colors = np.array([trimesh.visual.random_color()
                                 for i in points])
        cloud_boundary.vertices_color = cloud_colors
        scene_list.append(cloud_boundary)
    scene = trimesh.Scene(scene_list)
    scene.show()

    # here is how to get the vertices that define the boundary of
    # a texture on a mesh
    mesh = sio.load_mesh('data/example_mesh.gii')
    tex_parcel = sio.load_texture('data/example_texture_parcel.gii')
    col_map = trimesh.visual.color.interpolate(tex_parcel.darray)
    mesh.visual.vertex_colors = col_map
    boundary = stop.texture_boundary(mesh, tex_parcel.darray[0], 0)
    print(boundary)
    scene_list = [mesh]
    for bound in boundary:
        path_visual = trimesh.load_path(mesh.vertices[bound])
        path_visual.vertices_color = trimesh.visual.random_color()
        scene_list.append(path_visual)
        # points = mesh.vertices[bound]
        # cloud_boundary = trimesh.points.PointCloud(points)
        # cloud_colors = np.array([trimesh.visual.random_color()
        #                          for i in points])
        # cloud_boundary.vertices_color = cloud_colors
        # scene_list.append(cloud_boundary)
    # boundary_vertices = stop.texture_boundary_vertices(tex_parcel.darray, 0,
    # mesh.vertex_neighbors)
    # print(boundary_vertices)
    # path_visual = trimesh.load_path(mesh.vertices[boundary[3]])
    # create a scene with the mesh, path, and points
    # scene = trimesh.Scene([path_visual, mesh ])

    # trimesh.points.plot_points(points)
    # plt.show()

    # submeshes = stop.cut_mesh(mesh, tex_parcel.darray)
    # #splt.pyglet_plot(submeshes[0])
    # cloud_boundary = trimesh.points.PointCloud(mesh.vertices[boundary])
    # scene = trimesh.Scene([cloud_boundary, mesh])
    # scene = trimesh.Scene([mesh, cloud_boundary])

    col = mesh.visual.vertex_colors
    col[:, 3] = 100
    mesh.visual.vertex_colors = col

    scene_list.append(path_visual)
    print(path_visual)
    scene = trimesh.Scene(scene_list)
    scene.show(smooth=False)

    sub_meshes, sub_tex, sub_corresp = stop.cut_mesh(mesh, tex_parcel.darray[0])
    scene_list = list()
    for s_mesh in sub_meshes:
        print(s_mesh)
        s_mesh.visual.vertex_colors = trimesh.visual.random_color()
        scene_list.append(s_mesh)
    scene = trimesh.Scene(scene_list)
    scene.show(smooth=False)
