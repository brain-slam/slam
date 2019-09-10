import slam.plot as splt
import slam.io as sio
import slam.geodesics as sgeo
import numpy as np
import trimesh

if __name__ == '__main__':

    mesh = sio.load_mesh('data/example_mesh.gii')

    vert_id = 0
    max_geodist = 10

    geo_distance = sgeo.compute_gdist(mesh, vert_id)
    splt.pyglet_plot(mesh, geo_distance, plot_colormap=True)
    # get the vertices index in specified geo_distance of vert
    area_geodist_vi = np.where(geo_distance < max_geodist)[0]
    print(area_geodist_vi)

    area_geodist = sgeo.local_gdist_matrix(mesh, max_geodist)
    splt.pyglet_plot(mesh, area_geodist[0].toarray().squeeze(),
                     plot_colormap=True)
    # print(area_geodist[0].toarray()-geo_distance)

    # print the vertex index
    for i in range(mesh.vertices.shape[0]):
        vert_distmap = area_geodist[i].toarray()[0]
        area_geodist_v = np.where(vert_distmap > 0)[0]
        print(area_geodist_v)
    #

    # arbitrary indices of mesh.vertices to test with
    start = 0
    end = int(len(mesh.vertices) / 2.0)
    path = sgeo.shortest_path(mesh, start, end)
    print(path)

    # VISUALIZE RESULT
    # make the sphere transparent-ish
    mesh.visual.face_colors = [100, 100, 100, 100]
    # Path3D with the path between the points
    path_visual = trimesh.load_path(mesh.vertices[path])
    print(path_visual)
    # visualizable two points
    points_visual = trimesh.points.PointCloud(mesh.vertices[[start, end]])

    # create a scene with the mesh, path, and points
    scene = trimesh.Scene([
        points_visual,
        path_visual,
        mesh])

    scene.show(smooth=False)
