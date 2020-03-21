"""
.. _example_basics:

===================================
Show basic use of slam
===================================
"""

# Authors: Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2

# import numpy as np
# import slam.io as sio
#
#
#
# '''
# This scripts shows examples of basic functionalities offered by SLAM.
# Some (most) of these are actually inherited from Trimesh
# This script does not plot anything, see example_plot.py for that purpose
# '''
# mesh_file = 'data/example_mesh.gii'
# # loading a mesh stored on the disc as a gifti file,
# # this is a feature of SLAM
# mesh = sio.load_mesh(mesh_file)
#
# # affine transformations can be applied to mesh objects
# mesh.apply_transform(mesh.principal_inertia_transform)
# # laplacian smoothing is available in Trimesh
# # mesh_s = sm.filter_laplacian(mesh, iterations=20)
#
# # mesh.fill_holes() is able to fill missing face but do not handle
# # larger holes
#
# # interesting properties / functions of a mesh
# # see base.py for more details
# # what's the euler number for the mesh?
# print('mesh.euler_number=', mesh.euler_number)
#
# # access mesh edges
# mesh.edges
# # access mesh faces
# mesh.faces
# # access mesh vertice
# mesh.vertices
# # access mesh edges
# mesh.edges
#
# # what's the area of the mesh
# print('mesh.area=', mesh.area)
#
# # compute the area of each face
# mesh.area_faces
# # access mesh faces angles
# mesh.face_angles
# # access mesh volume
# mesh.volume
#
# # get the face_normal of the mesh
# mesh.face_normals
#
# # get the vertex_normals of the mesh
# mesh.vertex_normals
#
# # access mesh vertex connectivity
# mesh.vertex_neighbors
#
# # compute mesh convex hull
# c_h_mesh = mesh.convex_hull
#
# # kdtree of the vertices
# # see example_kdtree.py
#
# # mesh refinement by subdivision of face
# mesh.subdivide()
#
# # extract 100 mesh vertices picked at random
# mesh.sample(100)
#
# # voxelize the mesh
# mesh.voxelized(2)
#
# # boundary of the mesh or list of faces
# # mesh.outline()
#
# # the convex hull is another Trimesh object that is available as a property
# # lets compare the volume of our mesh with the volume of its convex hull
# np.divide(mesh.volume, mesh.convex_hull.volume)
#
# # since the mesh is watertight, it means there is a
# # volumetric center of mass which we can set as the origin for our mesh
# mesh.vertices -= mesh.center_mass
#
# # what's the moment of inertia for the mesh?
# mesh.moment_inertia
#
# # if there are multiple bodies in the mesh we can split the mesh by
# # connected components of face adjacency
# # since this example mesh is a single watertight body we get a list of one
# # mesh
# # mesh.split()
#
# # find groups of coplanar adjacent faces
# # facets, facets_area = mesh.facets(return_area=True)
#
# # transform method can be passed a (4,4) matrix and will cleanly apply the
# # transform
# mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
#
# # axis aligned bounding box is available
# mesh.bounding_box.extents
#
# # a minimum volume oriented bounding box also available
# # primitives are subclasses of Trimesh objects which automatically generate
# # faces and vertices from data stored in the 'primitive' attribute
# mesh.bounding_box_oriented.primitive.extents
# mesh.bounding_box_oriented.primitive.transform
#
# # bounding spheres and bounding cylinders of meshes are also
# # available, and will be the minimum volume version of each
# # except in certain degenerate cases, where they will be no worse
# # than a least squares fit version of the primitive.
# print(mesh.bounding_box_oriented.volume,
#       mesh.bounding_cylinder.volume,
#       mesh.bounding_sphere.volume)
