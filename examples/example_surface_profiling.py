import slam.surface_profiling as surfpf
import slam.io as sio
import trimesh
import trimesh.visual.color
import numpy as np


if __name__ == '__main__':

    mesh = sio.load_mesh('data/example_mesh.gii')


    vert_index = 1000  # a given vertex index
    vert0 = mesh.vertices[vert_index]
    norm0 = mesh.vertex_normals[vert_index]

    theta = 10  # rotation angle
    r = 0.1    # sampling step length
    m = 45     # max sampling distance

    profile_points = surfpf.surface_profiling_vert(vert0, norm0, theta, r, m, mesh)

    # VISUALIZE RESULT
    prof_points_mesh = profile_points.reshape(profile_points.shape[0] * profile_points.shape[1], 3)
    prof_points_colors = np.zeros(prof_points_mesh.shape)
    points_mesh = trimesh.points.PointCloud(prof_points_mesh, colors=np.array(prof_points_colors, dtype=np.uint8))

    # set the points colors
    red, yellow, green, blue = [[255, 0, 0, 255], [255, 255, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]
    color_i = np.zeros(4)
    for i in range(len(prof_points_mesh)):
        degree = i / m
        num_profiles = int(360 / theta)
        segment_color = num_profiles / 3

        if degree <= segment_color:
            color_i = trimesh.visual.color.linear_color_map(degree / segment_color, [red, yellow])
        elif segment_color < degree <= segment_color*2:
            color_i = trimesh.visual.color.linear_color_map(degree / segment_color - 1, [yellow, green])
        elif segment_color*2 < degree <= num_profiles:
            color_i = trimesh.visual.color.linear_color_map(degree / segment_color - 2, [green, blue])
        points_mesh.colors[i] = np.array(color_i)

    # print(profile_points)

    # create a scene with the mesh and profiling points
    scene = trimesh.Scene([points_mesh, mesh])
    scene.show(smooth=False)

