import numpy as np
from scipy.sparse.linalg import lgmres
import slam.differential_geometry as sdg
import slam.curvature as scurv
import slam.utils as sutl

########################
# error tolerance for lgmres solver
solver_tolerance = 1e-6
########################


def depth_potential_function(mesh, alphas=None, curvature=None):
    """
    compute the depth potential function of a mesh as desribed in
    Boucher, M., Whitesides, S., & Evans, A. (2009).
    Depth potential function for folding pattern representation,
    registration and analysis.
    Medical Image Analysis, 13(2), 203–14.
    doi:10.1016/j.media.2008.09.001
    :param mesh:
    :param curvature:
    :param alphas:
    :return:
    """
    if alphas is None:
        alphas = [0.03]
    if curvature is None:
        # Comptue mean curvature from principal curvatures
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 \
            = scurv.curvatures_and_derivatives(mesh)
        mean_curv = (
                0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :]))
        filt_mean_curv = (
            sutl.z_score_filtering(np.array([mean_curv]), z_thresh=3))
        curvature = filt_mean_curv[0]

    L, LB = sdg.compute_mesh_laplacian(mesh, lap_type="fem")
    B = (
        2
        * LB
        * (curvature - (np.sum(curvature * LB.diagonal())
                        / np.sum(LB.diagonal())))
    )
    # be careful with factor 2 used in eq (13)

    dpf = []
    for ind, alpha in enumerate(alphas):
        M = alpha * LB + L / 2
        dpf_t, info = lgmres(M.tocsr(), B, rtol=solver_tolerance)
        dpf.append(dpf_t)

    return dpf


def dpf_star(mesh, alphas=None, adaptation=None, curvature=None):
    """
    compute the depth potential function of a mesh. The scale of
    interest is adapted to the size of the mesh.
    :param mesh: TRIMESH MESH
    :param curvature: TEXTURE (darray)
    :param alphas: LIST : adapt the size of interest for computing
    sulcal depth
    :return: TEXTURE (darray)

    Parameters
    ----------
    adaptation
    """
    if alphas is None:
        alphas = [500]
    if curvature is None:
        # Comptue mean curvature from principal curvatures
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 \
            = scurv.curvatures_and_derivatives(mesh)
        mean_curv = (
                0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :]))
        filt_mean_curv = (
            sutl.z_score_filtering(np.array([mean_curv]), z_thresh=3))
        curvature = filt_mean_curv[0]
    if adaptation is None:
        adaptation = 'volume_hull'
    # print(source.shape)
    # print(mesh.vertices.shape)
    # adaptation of the scale of interest
    if adaptation == 'volume_hull':
        hull = mesh.convex_hull
        vol_hull = hull.volume
        lc = np.power(vol_hull, 1/3)
    if adaptation == 'volume':
        vol = mesh.volume
        lc = np.power(vol, 1/3)
    if adaptation == 'surface':
        surface = mesh.area
        lc = np.power(surface, 1/2)

    # compute the laplacian
    L, LB = sdg.compute_mesh_laplacian(mesh)

    # dedimensialisation of the laplacian and the curvature
    L = L * np.square(lc)
    curvature = curvature * lc

    # compute the dpf
    B = (
        LB
        * (curvature - (np.sum(curvature * LB.diagonal())
                        / np.sum(LB.diagonal()))))

    dpf = []
    for ind, alpha in enumerate(alphas):
        M = (alpha * LB) + L
        dpf_t, info = lgmres(M.tocsr(), B, rtol=solver_tolerance)
        dpf.append(dpf_t)

    return dpf
