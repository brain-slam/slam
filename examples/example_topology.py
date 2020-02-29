import slam.io as sio
import slam.topology as stop
import slam.plot as splt
import slam.generate_parametric_surfaces as sps
import numpy as np
from vispy.scene import Line
from visbrain.objects import VispyObj, SourceObj


if __name__ == '__main__':

    # mesh_boundary works fine
    # here is how to get the vertices that define the boundary of an open mesh
    K = [-1, -1]
    open_mesh = sps.generate_quadric(K, nstep=5)
    print('================= mesh_boundary =================')
    print('Identify the vertices lying on the boundary of the mesh and order'
          'them to get a path traveling across boundary vertices')
    print('The output is a list of potentially more than one boudaries '
          'depending on the topology of the input mesh')
    open_mesh_boundary = stop.mesh_boundary(open_mesh)
    print(open_mesh_boundary)
    print('Here the mesh has a single boundary')

    # WARNING : BrainObj should be added first before
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
    visb_sc.preview()

    # # # here is how to get the vertices that define the boundary of
    # # # a texture on a mesh
    mesh = sio.load_mesh('data/example_mesh.gii')
    tex_parcel = sio.load_texture('data/example_texture_parcel.gii')
    # # bound_verts =
    # stop.texture_boundary_vertices(
    # tex_parcel.darray[0], 20, mesh.vertex_neighbors)
    # boundaries = list()
    # for val in np.unique(tex_parcel.darray[0]):
    #     boundary = stop.texture_boundary(mesh, tex_parcel.darray[0], val)
    #     boundaries.append(boundary)
    #
    # # plot
    # visb_sc2 = splt.visbrain_plot(mesh=mesh, tex=tex_parcel.darray[0],
    #                               caption='texture boundary')
    cols = ['red', 'green', 'yellow', 'blue']
    # ind=0
    # for bound in boundaries:
    #     for sub_bound in bound:
    #         s_rad = SourceObj('rad', mesh.vertices[sub_bound],
    #         color=cols[ind], symbol='square',
    #                           radius_min=10)
    #         visb_sc2.add_to_subplot(s_rad)
    #         lines = Line(pos=mesh.vertices[sub_bound],
    #         color=cols[ind], width=10)
    #         # wrap the vispy object using visbrain
    #         l_obj = VispyObj('line', lines)
    #         visb_sc2.add_to_subplot(l_obj)
    #         ind+=1
    #         if ind==len(cols):
    #             ind=0
    #     #path_visual = trimesh.load_path(mesh.vertices[bound])
    #     #path_visual.vertices_color = trimesh.visual.random_color()
    #     # points = mesh.vertices[bound]
    #     # cloud_boundary = trimesh.points.PointCloud(points)
    #     # cloud_colors = np.array([trimesh.visual.random_color()
    #     #                          for i in points])
    #     # cloud_boundary.vertices_color = cloud_colors
    #     # scene_list.append(cloud_boundary)
    # visb_sc2.preview()

    # cut_mesh works fine!!
    print('================= cut_mesh =================')
    print('Cut he mesh into subparts corresponding to the different values in '
          'the texture tex_parcel')
    parc_u = np.unique(tex_parcel.darray[0])
    print('Here the texture contains {0} different values: {1}'
          ''.format(len(parc_u), parc_u))
    sub_meshes, sub_tex, sub_corresp = stop.cut_mesh(mesh,
                                                     tex_parcel.darray[0])
    print('The order of the resulting submeshes is given by'
          ' the second ouput: {}'.format(sub_tex))
    print('The respective indices of the vertices of each submesh in the '
          'original mesh before the cut is given by the third output:')
    print(sub_corresp)

    scene_list = list()

    cuted_mesh = sub_meshes[-1]
    # joint_mesh = sub_meshes[0] + sub_meshes[1]
    # joint_tex = np.ones((joint_mesh.vertices.shape[0],))
    # joint_tex[:sub_meshes[0].vertices.shape[0]] = 10
    joint_mesh = sub_meshes[0]
    joint_tex = np.zeros((sub_meshes[0].vertices.shape[0],))
    last_ind = sub_meshes[0].vertices.shape[0]
    for ind, sub_mesh in enumerate(sub_meshes[1:]):
        sub_tex = np.ones((sub_mesh.vertices.shape[0],)) * (ind + 1)
        joint_mesh += sub_mesh
        joint_tex = np.hstack((joint_tex, sub_tex))
    visb_sc2 = \
        splt.visbrain_plot(mesh=joint_mesh, tex=joint_tex,
                           caption='mesh parts shown in different colors')
    ind = 0
    boundaries = stop.mesh_boundary(cuted_mesh)
    for bound in boundaries:
        s_rad = \
            SourceObj('rad', cuted_mesh.vertices[bound],
                      color=cols[ind], symbol='square',
                      radius_min=10)
        visb_sc2.add_to_subplot(s_rad)
        lines = Line(pos=mesh.vertices[bound], color=cols[ind], width=10)
        # wrap the vispy object using visbrain
        l_obj = VispyObj('line', lines)
        visb_sc.add_to_subplot(l_obj)
        ind += 1
        if ind == len(cols):
            ind = 0

    visb_sc2.preview()

    # close_mesh works fine!!
    print('================= close_mesh =================')

    # mesh = sio.load_mesh('data/example_mesh.gii')
    # tex_parcel = sio.load_texture('data/example_texture_parcel.gii')
    # a_tex = tex_parcel.darray[0]
    # print(np.unique(a_tex))
    # #a_tex[a_tex<50]=0#.00005]=0
    # #a_tex[a_tex>51]=0
    # sub_meshes, sub_tex, sub_corresp = stop.cut_mesh(mesh,
    #                                                    a_tex)
    # mesh = sub_meshes[-1]
    # open_mesh_boundary = stop.mesh_boundary(cuted_mesh)
    mesh_closed, nb_verts_added = stop.close_mesh(cuted_mesh)
    # , [open_mesh_boundary[2]])

    print(mesh.is_watertight)
    print(mesh_closed.is_watertight)
    # mesh.show()
    # mesh_closed.show()

    visb_sc3 = splt.visbrain_plot(mesh=cuted_mesh, caption='open mesh')
    # create points with vispy
    for bound in boundaries:
        points = cuted_mesh.vertices[bound]
        s_rad = SourceObj('rad', points, color='blue', symbol='square',
                          radius_min=10)
        visb_sc3.add_to_subplot(s_rad)
        lines = Line(pos=cuted_mesh.vertices[bound], width=10, color='r')
        # wrap the vispy object using visbrain
        l_obj = VispyObj('line', lines)
        visb_sc3.add_to_subplot(l_obj)
    visb_sc3 = splt.visbrain_plot(mesh=mesh_closed,
                                  caption='closed mesh',
                                  visb_sc=visb_sc3)
    visb_sc3.preview()
