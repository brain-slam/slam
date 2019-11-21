import trimesh
import slam.io as sio
import slam.topology as stop
import slam.plot as splt
import slam.generate_parametric_surfaces as sps
import numpy as np

if __name__ == '__main__':
    # here is how to get the vertices that define the boundary of an open mesh
    K = [1, 1]
    open_mesh = sps.generate_quadric(K, nstep=20)

    open_mesh_boundary = stop.mesh_boundary(open_mesh)
    print(open_mesh_boundary)

    from vispy.scene import Line
    from visbrain.objects import VispyObj, SourceObj
    # WARNING : BrainObj should be added first before
    visb_sc = splt.visbrain_plot(mesh=open_mesh, caption='open mesh')
    # create points with vispy
    for bound in open_mesh_boundary:
        points = open_mesh.vertices[bound]
        s_rad = SourceObj('rad', points, color='red', symbol='square')
        visb_sc.add_to_subplot(s_rad)
    visb_sc.preview()
    # here is how to get the vertices that define the boundary of
    # a texture on a mesh
    mesh = sio.load_mesh('data/example_mesh.gii')
    tex_parcel = sio.load_texture('data/example_texture_parcel.gii')
    col_map = trimesh.visual.color.interpolate(tex_parcel.darray)
    mesh.visual.vertex_colors = col_map
    boundary = stop.texture_boundary(mesh, tex_parcel.darray[0], 0)
    print(boundary)
    scene_list = [mesh]
    visb_sc2 = splt.visbrain_plot(mesh=mesh, tex=tex_parcel.darray[0],
                                  caption='texture boundary')
    for bound in boundary:
        lines = Line(pos=mesh.vertices[bound], )
        # wrap the vispy object using visbrain
        l_obj = VispyObj('line', lines)
        visb_sc2.add_to_subplot(l_obj)
        path_visual = trimesh.load_path(mesh.vertices[bound])
        path_visual.vertices_color = trimesh.visual.random_color()
        scene_list.append(path_visual)
        # points = mesh.vertices[bound]
        # cloud_boundary = trimesh.points.PointCloud(points)
        # cloud_colors = np.array([trimesh.visual.random_color()
        #                          for i in points])
        # cloud_boundary.vertices_color = cloud_colors
        # scene_list.append(cloud_boundary)
    visb_sc2.preview()
    # boundary_vertices = stop.texture_boundary_vertices(tex_parcel.darray, 0,
    # mesh.vertex_neighbors)
    # print(boundary_vertices)
    # path_visual = trimesh.load_path(mesh.vertices[boundary[3]])
    # create a scene with the mesh, path, and points
    # scene = trimesh.Scene([path_visual, mesh ])

    # trimesh.points.plot_points(points)
    # plt.show()

    # col = mesh.visual.vertex_colors
    # col[:, 3] = 100
    # mesh.visual.vertex_colors = col
    # scene_list.append(path_visual)
    # print(path_visual)
    # scene = trimesh.Scene(scene_list)
    # scene.show(smooth=False)

    sub_meshes, sub_tex, sub_corresp = stop.cut_mesh(mesh,
                                                     tex_parcel.darray[0])
    scene_list = list()
    joint_mesh = sub_meshes[0]+sub_meshes[1]
    joint_tex = np.ones((joint_mesh.vertices.shape[0],))
    joint_tex[:sub_meshes[0].vertices.shape[0]] = 10
    visb_sc = splt.visbrain_plot(mesh=joint_mesh, tex=joint_tex,
                                 caption='mesh parts shown in'
                                         ' different colors')
    visb_sc.preview()
