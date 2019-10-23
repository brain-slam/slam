import numpy as np
from scipy import sparse
import scipy.stats.stats as sss
from scipy.sparse.linalg import lgmres, eigsh
import trimesh
########################
# error tolerance for lgmres solver
solver_tolerance = 1e-6
########################


def mesh_laplacian_eigenvectors(mesh, nb_vectors=1):
    """
    compute the nb_vectors eigenvectors of the graph Laplacian of mesh
    :param mesh:
    :param nb_vectors:
    :return:
    """
    lap, lap_b = compute_mesh_laplacian(mesh, lap_type='fem')
    w, v = eigsh(lap.tocsr(), nb_vectors + 1, M=lap_b.tocsr(),
                 sigma=solver_tolerance)
    return v[:, 1:]


# def mesh_fiedler_length(mesh, dist_type='geodesic', fiedler=None):
#     """
#     distance between the two vertices corresponding to the min and max of
# the 2d laplacien eigen vector
#     :param mesh:
#     :param dist_type:
#     :param fiedler:
#     :return:
#     """
#     if fiedler is None:
#         fiedler = mesh_laplacian_eigenVectors(mesh, 1)
#     imin = fiedler.argmin()
#     imax = fiedler.argmax()
#     vert = np.array(mesh.vertex())
#
#     if dist_type == 'geodesic':
#         print('Computing GEODESIC distance between the max and min')
#         g = aims.GeodesicPath(mesh, 3, 0)
#         dist = g.shortestPath_1_1_len(int(imin), int(imax))
#
#     else:
#         print('Computing EUCLIDIAN distance between the max and min')
#         min_max = vert[imin, :]-vert[imax, :]
#         dist = np.sqrt(np.sum(min_max * min_max, 0))
#     return(dist, fiedler)


def laplacian_mesh_smoothing(mesh, nb_iter, dt):
    """
    smoothing the mesh by solving the heat equation using fem Laplacian
    ADD REF
    :param mesh:
    :param nb_iter:
    :param dt:
    :return:
    """
    print('    Smoothing mesh')
    lap, lap_b = compute_mesh_laplacian(mesh, lap_type='fem')
    smoothed_vert = laplacian_smoothing(mesh.vertices, lap, lap_b, nb_iter, dt)
    return trimesh.Trimesh(faces=mesh.faces, vertices=smoothed_vert,
                           metadata=mesh.metadata, process=False)


def laplacian_texture_smoothing(mesh, tex, nb_iter, dt):
    """
    smoothing the texture by solving the heat equation using fem Laplacian
    :param mesh:
    :param tex:
    :param nb_iter:
    :param dt:
    :return:
    """
    print('    Smoothing texture')
    lap, lap_b = compute_mesh_laplacian(mesh, lap_type='fem')
    return laplacian_smoothing(tex, lap, lap_b, nb_iter, dt)


def laplacian_smoothing(texture_data, lap, lap_b, nb_iter, dt):
    """
    sub-function for smoothing using fem Laplacian
    :param texture_data:
    :param lap:
    :param lap_b:
    :param nb_iter:
    :param dt:
    :return:
    """
    mod = 1
    if nb_iter > 10:
        mod = 10
    if nb_iter > 100:
        mod = 100
    if nb_iter > 1000:
        mod = 1000
    # print(tex.shape[0])
    # print(tex.ndim)
    # if tex.ndim < 2:
    #    Mtex = tex.reshape(tex.shape[0],1)
    # else:
    #    Mtex = tex
    # using Implicit scheme
    # B(X^(n+1)-X^n)/dt+L(X^(n+1))=0
    M = lap_b + dt * lap
    for i in range(nb_iter):
        texture_data = lap_b * texture_data
        if texture_data.ndim > 1:
            for d in range(texture_data.shape[1]):
                texture_data[:, d], infos = lgmres(M.tocsr(),
                                                   texture_data[:, d],
                                                   tol=solver_tolerance)
        else:
            texture_data, infos = lgmres(M.tocsr(), texture_data,
                                         tol=solver_tolerance)
        if i % mod == 0:
            print(i)

    # using Explicit scheme, convergence guaranteed only for dt<1 and not
    # faster than implicit when using fem Laplacian
    # B(X^(n+1)-X^n)/dt+L(X^n)=0
    # M = B-dt*L
    # for i in range(Niter):
    #     Mtex = M * Mtex
    #     Mtex, infos = lgmres(B.tocsr(), Mtex, tol=solver_tolerance)
    #     if (i % mod == 0):
    #         print(i)
    print('    OK')
    return texture_data


def compute_mesh_weights(mesh, weight_type='conformal', cot_threshold=None,
                         z_threshold=None):
    """
    compute a weight matrix
    W is sparse weight matrix and W(i,j) = 0 is vertex i and vertex j are not
    connected in the mesh.

    type is either
        'combinatorial': W(i,j) = 1 is vertex i is conntected to vertex j.
        'distance': W(i,j) = 1/d_ij^2 where d_ij is distance between vertex
            i and j.
        'conformal': W(i,j) = cot(alpha_ij)+cot(beta_ij) where alpha_ij and
            beta_ij are the adjacent angle to edge (i,j)
    If options.normalize = 1, the the rows of W are normalize to sum to 1.
    :param mesh:
    :param weight_type:
    :param cot_threshold:
    :param z_threshold:
    :return:
    """
#    cot_threshold=0.00001
#   print('angle threshold')
    print('    Computing mesh weights')
    vert = mesh.vertices
    poly = mesh.faces

    Nbv = vert.shape[0]
    W = sparse.lil_matrix((Nbv, Nbv))
    femB = sparse.lil_matrix((Nbv, Nbv))
    if weight_type == 'conformal' or weight_type == 'fem':
        threshold = 0.0001  # np.spacing(1)??
        threshold_needed = 0
        for i in range(3):
            i1 = np.mod(i, 3)
            i2 = np.mod(i + 1, 3)
            i3 = np.mod(i + 2, 3)
            pp = vert[poly[:, i2], :] - vert[poly[:, i1], :]
            qq = vert[poly[:, i3], :] - vert[poly[:, i1], :]
            cr = np.cross(pp, qq)
            area = np.sqrt(np.sum(np.power(cr, 2), 1)) / 2
#             nopp = np.apply_along_axis(np.linalg.norm, 1, pp)
#             noqq = np.apply_along_axis(np.linalg.norm, 1, qq)
            noqq = np.sqrt(np.sum(qq * qq, 1))
            nopp = np.sqrt(np.sum(pp * pp, 1))
            thersh_nopp = np.where(nopp < threshold)[0]
            thersh_noqq = np.where(noqq < threshold)[0]
            if len(thersh_nopp) > 0:
                nopp[thersh_nopp] = threshold
                threshold_needed += len(thersh_nopp)
            if len(thersh_noqq) > 0:
                noqq[thersh_noqq] = threshold
                threshold_needed += len(thersh_noqq)
    #        print(np.min(noqq))
            pp = pp / np.vstack((nopp, np.vstack((nopp, nopp)))).transpose()
            qq = qq / np.vstack((noqq, np.vstack((noqq, noqq)))).transpose()
            ang = np.arccos(np.sum(pp * qq, 1))
            # ############## preventing infs in weights
            inds_zeros = np.where(ang == 0)[0]
            ang[inds_zeros] = threshold
            threshold_needed_angle = len(inds_zeros)
            ################################
            cot = 1 / np.tan(ang)
            if cot_threshold is not None:
                thresh_inds = cot < 0
                cot[thresh_inds] = cot_threshold
                threshold_needed_angle += np.count_nonzero(thresh_inds)
            W = W + sparse.coo_matrix((cot, (poly[:, i2], poly[:, i3])),
                                      shape=(Nbv, Nbv))
            W = W + sparse.coo_matrix((cot, (poly[:, i3], poly[:, i2])),
                                      shape=(Nbv, Nbv))
            femB = femB + sparse.coo_matrix((area / 12,
                                             (poly[:, i2], poly[:, i3])),
                                            shape=(Nbv, Nbv))
            femB = femB + sparse.coo_matrix((area / 12,
                                             (poly[:, i3], poly[:, i2])),
                                            shape=(Nbv, Nbv))

        # if weight_type == 'fem' :
        #     W.data = W.data/2

        nnz = W.nnz
        if z_threshold is not None:
            z_weights = sss.zscore(W.data)
            inds_out = np.where(np.abs(z_weights) > z_threshold)[0]
            W.data[inds_out] = np.mean(W.data)
            print('    -Zscore threshold needed for ', len(inds_out),
                  ' values = ', 100 * len(inds_out) / nnz, ' %')
            # inds_out_inf = np.where(z_weights < -z_thresh)[0]
            # inds_out_sup = np.where(z_weights > z_thresh)[0]
            # val_inf = np.max(W.data[inds_out_inf])
            # W.data[inds_out_inf] = val_inf
            # val_sup = np.min(W.data[inds_out_sup])
            # W.data[inds_out_sup] = val_sup
            # print('    -Zscore threshold needed for ',
            # len(inds_out_inf)+len(inds_out_sup),' values-')
        print('    -edge length threshold needed for ', threshold_needed,
              ' values = ', 100 * threshold_needed / nnz, ' %')
        if cot_threshold is not None:
            print('    -cot threshold needed for ', threshold_needed_angle,
                  ' values = ', 100 * threshold_needed_angle / nnz, ' %')

    li = np.hstack(W.data)
    nb_Nan = len(np.where(np.isnan(li))[0])
    nb_neg = len(np.where(li < 0)[0])
    print('    -number of Nan in weights: ',
          nb_Nan, ' = ', 100 * nb_Nan / nnz, ' %')
    print('    -number of Negative values in weights: ',
          nb_neg, ' = ', 100 * nb_neg / nnz, ' %')

    return W.tocsr(), femB.tocsr()


def compute_mesh_laplacian(mesh, weights=None, fem_b=None,
                           lap_type='conformal'):
    """
    compute laplacian of a mesh
    :param mesh:
    :param weights:
    :param fem_b:
    :param lap_type:
    :return:
    """
    print('    Computing Laplacian')
    if weights is None:
        (weights, fem_b) = compute_mesh_weights(mesh, weight_type=lap_type)

    if lap_type == 'fem':
        weights.data = weights.data / 2

    N = weights.shape[0]
    sB = fem_b.sum(axis=0)
    diaB = sparse.dia_matrix((sB, 0), shape=(N, N))
    B = sparse.lil_matrix(diaB + fem_b)
    s = weights.sum(axis=0)
    dia = sparse.dia_matrix((s, 0), shape=(N, N))
    L = sparse.lil_matrix(dia - weights)

    li = np.hstack(L.data)
    print('    -nb Nan in L : ', len(np.where(np.isnan(li))[0]))
    print('    -nb Inf in L : ', len(np.where(np.isinf(li))[0]))

    return L, B


def depth_potential_function(mesh, curvature, alphas):
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
    L, LB = compute_mesh_laplacian(mesh, lap_type='fem')
    B = -LB * (curvature - (np.sum(curvature * LB.diagonal())
                            / np.sum(LB.diagonal())))

    dpf = []
    for ind, alpha in enumerate(alphas):
        M = alpha * LB + L / 2
        dpf_t, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
        dpf.append(dpf_t)

    ############################
    # old, slower and less accurate implementation using conformal laplacian
    # instead of fem
    ############################
    # vert_voronoi = vertexVoronoi(mesh)
    # L, LB = compute_mesh_laplacian(mesh, lap_type='conformal')
    # B = -2 * vert_voronoi * (curvature-( np.sum(curvature*vert_voronoi)
    # / vert_voronoi.sum() ))
    # B=B.squeeze()
    # for ind, alpha in enumerate(alphas):
    #     A = sparse.dia_matrix((alpha*vert_voronoi, 0), shape=(Nbv, Nbv))
    #     M = A+L
    #     dpf_t, info = lgmres(M.tocsr(), B, tol=solver_tolerance)
    #     dpf.append(dpf_t)
    return dpf
