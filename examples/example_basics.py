import trimesh
import numpy as np
import slam.io as sio
import slam.plot as splt
import nibabel as nb
from trimesh import smoothing as sm

if __name__ == '__main__':

    mesh = sio.load_mesh('example_mesh.gii')
    mesh.apply_transform(mesh.principal_inertia_transform)
    curv_gifti = nb.gifti.read('example_texture.gii')
    curv_tex = curv_gifti.darrays[0].data.squeeze()

    splt.pyglet_plot(mesh, curv_tex)

    mesh_s = sm.filter_laplacian(mesh, iterations=20)
    mesh_s.show()

    # mesh.fill_holes()
    mesh.edges
    # interesting properties / functions of a mesh
    # see base.py for more details
    # what's the euler number for the mesh?
    print(mesh.euler_number)

    # access mesh faces
    mesh.faces
    # access mesh vertice
    mesh.vertices
    # access mesh edges
    mesh.edges

    # what's the area of the mesh
    print(mesh.area)

    # compute the area of each face
    mesh.area_faces
    # access mesh faces angles
    mesh.face_angles
    # access mesh volume
    mesh.volume

    # get the face_normal of the mesh
    mesh.face_normals

    # get the vertex_normals of the mesh
    mesh.vertex_normals

    # access mesh vertex connectivity
    mesh.vertex_neighbors

    # mesh convex hull
    c_h_mesh = mesh.convex_hull

    mesh.convex_hull.show()

    # close mesh holes
    mesh.fill_holes()

    # register to another using iterative ICP initiated by principal axes of
    # intertia
    # mesh.register(other)

    m_smooth = mesh.smoothed()

    mesh.show()
    m_smooth.show()

    # kdtree of the vertices
    # mesh.kdtree()

    # mesh refinement by subdivision of face
    mesh.subdivide()

    # extract 100 mesh vertices picked at random
    mesh.sample(100)

    # apply rigid transormation
    # mesh.apply_transform()

    mesh.voxelized(2).show()

    # boundary of the mesh or list of faces
    # mesh.outline()

    # the convex hull is another Trimesh object that is available as a property
    # lets compare the volume of our mesh with the volume of its convex hull
    np.divide(mesh.volume, mesh.convex_hull.volume)

    # since the mesh is watertight, it means there is a
    # volumetric center of mass which we can set as the origin for our mesh
    mesh.vertices -= mesh.center_mass

    # what's the moment of inertia for the mesh?
    mesh.moment_inertia

    # if there are multiple bodies in the mesh we can split the mesh by
    # connected components of face adjacency
    # since this example mesh is a single watertight body we get a list of one
    # mesh
    # mesh.split()

    # find groups of coplanar adjacent faces
    # facets, facets_area = mesh.facets(return_area=True)

    # transform method can be passed a (4,4) matrix and will cleanly apply the
    # transform
    mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

    # axis aligned bounding box is available
    mesh.bounding_box.extents

    # a minimum volume oriented bounding box also available
    # primitives are subclasses of Trimesh objects which automatically generate
    # faces and vertices from data stored in the 'primitive' attribute
    mesh.bounding_box_oriented.primitive.extents
    mesh.bounding_box_oriented.primitive.transform

    # show the mesh appended with its oriented bounding box
    # the bounding box is a trimesh.primitives.Box object, which subclasses
    # Trimesh and lazily evaluates to fill in vertices and faces when requested
    # (press w in viewer to see triangles)
    (mesh + mesh.bounding_box_oriented).show()

    # bounding spheres and bounding cylinders of meshes are also
    # available, and will be the minimum volume version of each
    # except in certain degenerate cases, where they will be no worse
    # than a least squares fit version of the primitive.
    print(mesh.bounding_box_oriented.volume,
          mesh.bounding_cylinder.volume,
          mesh.bounding_sphere.volume)
