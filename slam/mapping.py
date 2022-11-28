import numpy as np
import trimesh
from scipy.sparse.linalg import lgmres

# from scipy import sparse as ssp

import slam.differential_geometry as sdg
import slam.distortion as sdst
import slam.topology as stop


########################
# error tolerance for lgmres solver
solver_tolerance = 1e-6
########################


def spherical_mapping(
    mesh,
    mapping_type="laplacian_eigenvectors",
    conformal_w=1,
    authalic_w=1,
    dt=0.01,
    nb_it=10,
):
    """
    Computes the mapping between the input mesh and a sphere centered on 0 and of radius=1
    Several methods are implemented:
    -for laplacian eigenvectors, see:
    J. Lefèvre and G. Auzias, "Spherical Parameterization for Genus Zero Surfaces Using
    Laplace-Beltrami Eigenfunctions," in 2nd Conference on Geometric Science of
    Information, GSI, 2015, 121–29, https://doi.org/10.1007/978-3-319-25040-3_14.
    -for conformal mapping, see:
    Desbrun, M., Meyer, M., & Alliez, P., "Intrinsic parameterizations of surface meshes",
    Computer Graphics Forum, 21(3), 2002, 209–218. https://doi.org/10.1111/1467-8659.00580
    -for authalic and combined methods, see:
    Rachel a Yotter, Paul M. Thompson, and Christian Gaser, “Algorithms to Improve the
    Reparameterization of Spherical Mappings of Brain Surface Meshes.,” Journal of
    Neuroimaging 21, no. 2 (April 2011): e134-47, https://doi.org/10.1111/j.1552-6569.2010.00484.x.
    or
    Ilja Friedel, Peter Schröder, and Mathieu Desbrun, “Unconstrained Spherical Parameterization”,
    Journal of Graphics, GPU, and Game Tools 12, no. 1 (2007): 17–26.
    :param mesh: Trimesh object, input mesh to be mapped onto a sphere
    :param mapping_type: string, type of mapping method, possible options are:
    'laplacian_eigenvectors', 'conformal', 'authalic' or 'combined'
    :param conformal_w: Float, weight of the conformal constraint for the 'combined' method
    :param authalic_w: Float, weight of the authalic constraint for the 'combined' method
    :param dt: Float, discrtization step
    :param nb_it: Int, number of iterations
    :return: Trimesh object: the spherical representation of the input mesh, having the same
    adjacency (faces, edges, vertex indexing) as the input mesh.
    """
    # computing spherical mapping based on laplacian eigenvectors
    sph_vert = sdg.mesh_laplacian_eigenvectors(mesh, nb_vectors=3)
    norm_sph_vert = np.sqrt(np.sum(sph_vert * sph_vert, 1))
    sphere_vertices = sph_vert / np.tile(norm_sph_vert, (3, 1)).T

    if mapping_type == "laplacian_eigenvectors":
        return trimesh.Trimesh(
            faces=mesh.faces,
            vertices=sphere_vertices,
            metadata=mesh.metadata,
            process=False,
        )

    if mapping_type == 'conformal':
        L, B = sdg.compute_mesh_laplacian(mesh, lap_type='conformal')

    if mapping_type == 'authalic':
        L, B = sdg.compute_mesh_laplacian(mesh, lap_type='authalic')

    if mapping_type == 'combined':
        Lconf, Bconf = sdg.compute_mesh_laplacian(mesh, lap_type='conformal')
        Laut, Baut = sdg.compute_mesh_laplacian(mesh, lap_type='authalic')

        L = conformal_w * Lconf + authalic_w * Laut
    # continue the spherical mappig by minimizig the energy
    evol = list()
    for it in range(nb_it):
        sphere_vertices = sphere_vertices - dt * L.dot(sphere_vertices)
        # sphere_vertices * L
        norm_sph_vert = np.sqrt(np.sum(sphere_vertices * sphere_vertices, 1))
        sphere_vertices = sphere_vertices / np.tile(norm_sph_vert, (3, 1)).T
        if it % 10 == 0:
            sph = trimesh.Trimesh(
                faces=mesh.faces, vertices=sphere_vertices, process=False
            )
            angle_diff = sdst.angle_difference(sph, mesh)
            area_diff = sdst.area_difference(sph, mesh)
            edge_diff = sdst.edge_length_difference(sph, mesh)
            evol.append(
                [
                    np.sum(np.abs(angle_diff.flatten())),
                    np.sum(np.abs(area_diff.flatten())),
                    np.sum(np.abs(edge_diff.flatten())),
                ]
            )

    return trimesh.Trimesh(faces=mesh.faces,
                           vertices=sphere_vertices,
                           metadata=mesh.metadata, process=False), evol


def disk_conformal_mapping(
    mesh, lap_type="conformal", boundary=None, boundary_coords=None
):
    """
    Computes comformal mapping of a mesh to a disk, see the following references:
    Ulrich Pinkall and Konrad Polthier, “Computing Discrete Minimal Surfaces and
    Their Conjugates,” Experimental Mathematics, 1993, 1–33.
    and
    Mathieu Desbrun, Mark Meyer, and Pierre Alliez, “Intrinsic Parameterizations
    of Surface Meshes,” Computer Graphics Forum 21, no. 3 (2002): 209–18,
    https://doi.org/10.1111/1467-8659.00580.
    :param mesh: a trimesh object
    :param lap_type: type of mesh Laplacian to be used,
    see the function differential_geometry/compute_mesh_weights for more
    informations
    :param boundary: boundary of the mesh, resulting from the function
    topology/mesh_boundary
    :param boundary_coords: coordindates of the boundary vertices on the
    output disk, if None then uniform sampling
    :return: a trimesh object, planar disk representation of the input mesh
    """
    if boundary is None:
        boundary_t = stop.mesh_boundary(mesh)
        boundary = boundary_t[0]
    boundary = np.array(boundary)
    if boundary_coords is None:
        p = boundary.size
        t = np.arange(0, 2 * np.math.pi, (2 * np.math.pi / p))
        boundary_coords = np.array([np.cos(t), np.sin(t)])
    L, LB = sdg.compute_mesh_laplacian(mesh, lap_type=lap_type)
    Nv = len(mesh.vertices)  # np.array(mesh.vertex()).shape[0]
    print("Boundary Size:", boundary.shape)
    print("Laplacian Size:", L.shape)
    for i in boundary:
        L[i, :] = 0
        L[i, i] = 1
    L = L.tocsr()

    Rx = np.zeros(Nv)
    Ry = np.zeros(Nv)
    Rx[boundary] = boundary_coords[0, :]
    Ry[boundary] = boundary_coords[1, :]

    x, info = lgmres(L, Rx, tol=solver_tolerance)
    y, info = lgmres(L, Ry, tol=solver_tolerance)
    z = np.zeros(Nv)

    return trimesh.Trimesh(
        faces=mesh.faces,
        vertices=np.array([x, y, z]).T,
        metadata=mesh.metadata,
        process=False,
    )


def moebius_transformation(a, b, c, d, plane_mesh):
    """
    see https://en.wikipedia.org/wiki/M%C3%B6bius_transformation
    :param a:Complex
    :param b:Complex
    :param c:Complex
    :param d:Complex
    :param plane_mesh: trimesh mesh
    :return:
    """
    array_complex = plane_mesh.vertices[:,
                                        0] + 1.0j * plane_mesh.vertices[:, 1]
    numerator = (a * array_complex) + b
    denominator = (c * array_complex) + d

    transformed_complex_plane = numerator / denominator
    transformed_vertices = np.array(
        [
            transformed_complex_plane.real,
            transformed_complex_plane.imag,
            plane_mesh.vertices[:, 2],
        ]
    ).T.copy()
    transformed_plane_mesh = trimesh.Trimesh(
        vertices=transformed_vertices, faces=plane_mesh.faces.copy(), process=False
    )
    return transformed_plane_mesh


def stereo_projection(sphere_mesh, h=None, invert=True):
    """
    Computes the stereographic projection from the unit sphere (center = 0,
    radius = 1) onto the horizontal plane which 3rd coordinate is h of the
    vertices given
    :param sphere_mesh: trimesh spherical mesh to be projected onto the plane
    :param h: 3rd coordinate of the projection plane
    :param invert: Boolean value to invert output mesh faces orientation
    upwards
    :return: trimesh planar mesh, 3rd coordinate is equal to h
    """
    vertices = sphere_mesh.vertices.copy()
    if h is None:
        h = -1
    for ind, vert in enumerate(vertices):
        vertices[ind, 0] = (-h + 1) * vert[0] / (1 - vert[2])
        vertices[ind, 1] = (-h + 1) * vert[1] / (1 - vert[2])
        vertices[ind, 2] = h

    plane_mesh = trimesh.Trimesh(
        vertices=vertices, faces=sphere_mesh.faces.copy(), process=False
    )
    if invert:
        plane_mesh.invert()
    return plane_mesh


def inverse_stereo_projection(plane_mesh, h=None, invert=True):
    """
    Computes the inverse stereograhic projection from an horizontal plane onto
    the unit sphere (center = 0, radius = 1)
    :param plane_mesh: trimesh planar mesh to be inverse projected onto the
    sphere
    :param h: 3rd coordinate of the projection plane
    :param invert: Boolean value to invert input mesh faces orientation upwards
    to be consistent with stereo_projection
    :return: trimesh unit sphere from inverse projected plane_mesh
    """
    if invert:
        plane_mesh.invert()
    vertices = plane_mesh.vertices.copy()
    if h is None:
        h = vertices[0, 2]
    for ind, vert in enumerate(vertices):
        denom = (1 - h) ** 2 + vert[0] ** 2 + vert[1] ** 2
        vertices[ind, 2] = (-((1 - h) ** 2) + vert[0]
                            ** 2 + vert[1] ** 2) / denom
        vertices[ind, 1] = 2 * (1 - h) * vert[1] / denom
        vertices[ind, 0] = 2 * (1 - h) * vert[0] / denom

    sphere_mesh = trimesh.Trimesh(
        vertices=vertices, faces=plane_mesh.faces.copy(), process=False
    )
    return sphere_mesh
