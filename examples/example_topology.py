# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import trimesh
import numpy as np
import slam.io as sio
import slam.topology as stop
# import slam.plot as splt


if __name__ == '__main__':
    mesh = sio.load_mesh('/mnt/data/work/python_sandBox/slam/examples/'
                         'example_mesh.gii')
    tex_parcel = sio.load_texture('/mnt/data/work/python_sandBox/slam/'
                                  'examples/example_texture_parcel.gii')
    boundary = stop.texture_boundary(mesh, tex_parcel.darray, 0)
    print(boundary)
    scene_list = [mesh]
    for bound in boundary:
        points = mesh.vertices[bound]
        cloud_boundary = trimesh.points.PointCloud(points)
        cloud_colors = np.array([trimesh.visual.random_color()
                                 for i in points])
        cloud_boundary.vertices_color = cloud_colors
        scene_list.append(cloud_boundary)
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
    print([mesh, cloud_boundary])
    print(scene_list)
    scene = trimesh.Scene(scene_list)
    scene.show()
