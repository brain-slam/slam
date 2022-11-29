"""
Spectral Analysis of Gyrification
This module implements the Spectral Analysis described in D. Germanaud*, J. Lefevre*,
R. Toro, C. Fischer, J.Dubois, L. Hertz-Pannier, J.F. Mangin, Larger is twistier:
Spectral Analysis of Gyrification (SPANGY) applied to adult brain size polymorphism,
Neuroimage, 63 (3), 1257-1272, 2012.
"""
import numpy as np
import slam.differential_geometry as sdg
import time
from scipy.sparse.linalg import eigsh


def eigenpairs(mesh, nb_eig):
    """
    Parameters
    ----------
    mesh : Base.Trimesh

    nb_eig : Int
        number of eigenpairs to compute.
    Returns
    -------
    eigVal : Array of float
        eigenvalues computed.
    eigVects : Array of float
        eigenvectors computed.
    """
    lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='fem')
    tp1 = time.perf_counter()
    eigVal, eigVects = eigsh(lap.tocsr(), nb_eig, M=lap_b.tocsr(), sigma=1e-6)
    tp2 = time.perf_counter()

    print("    -computation time of eigenpairs : {} \n".format(tp2 - tp1))

    return eigVal, eigVects


def spangy_spectrum(f2analyse, MassMatrix, eigVec, eValues):
    """

    Parameters
    ----------
    f2analyse : Array of floats
        function to analyze (mean curvature).
    MassMatrix : Array of floats
        Used in the discretization of the eigenvalue problem.
    eigVec : Array of floats
        Eigenvectors.
    eValues : Array of floats
        Eigenvalues.
    Returns
    -------
    grouped_spectrum : Array of floats
        power in each spectral band.
    group_indices : Array of ints
        Indices of spectral bands.
    coefficients : Arrat of floats
        Fourier coefficients of the input function f2analyse.
    """
    coefficients = (f2analyse.dot(MassMatrix).dot(eigVec))[0]

    nlevels = int(0.5 * np.log(eValues[-1] / eValues[1]) / np.log(2))

    grouped_spectrum = np.zeros((nlevels + 2, 1));
    grouped_spectrum[0] = coefficients[0] ** 2;
    group_indices = np.zeros((nlevels + 2, 2), dtype=int)
    group_indices[0, :] = [0, 0]

    for k in range(nlevels):
        indice = np.where(eValues >= eValues[1] * 2 ** (2 * (k)))
        group_indices[k + 1, 0] = indice[0][0]
        indice = np.where(eValues <= eValues[1] * 2 ** (2 * (k + 1)))
        group_indices[k + 1, 1] = indice[0][-1]
        grouped_spectrum[k + 1] = np.sum(coefficients[group_indices[k + 1, 0]:group_indices[k + 1, 1] + 1] ** 2)

    group_indices[-1, 0] = group_indices[-2, 1] + 1
    group_indices[-1, 1] = eValues.size - 1
    grouped_spectrum[-1] = np.sum(coefficients[group_indices[-1, 0]:group_indices[-1, 1]] ** 2)

    return grouped_spectrum, group_indices, coefficients


def spangy_local_dominance_map(coefficients, f2analyse, nlevels, group_indices, eigVec):
    """
    Parameters
    ----------
    coefficients : TYPE
        DESCRIPTION.
    f2analyse : TYPE
        DESCRIPTION.
    nlevels : TYPE
        DESCRIPTION.
    group_indices : TYPE
        DESCRIPTION.
    eigVec : TYPE
        DESCRIPTION.
    Returns
    -------
    frecomposed : TYPE
        DESCRIPTION.
    f_ii : TYPE
        DESCRIPTION.
    """
    N = np.size(coefficients)
    frecomposed = np.zeros((nlevels), dtype='object')
    """band by band recomposition"""
    for i in range(0, nlevels):
        levels_i = np.arange(group_indices[i, 0], group_indices[i, 1] + 1, 1)
        f_ii = eigVec[:, N - levels_i - 1] * coefficients[levels_i - 1]
        frecomposed[i] = f_ii
    frecomposed = np.concatenate((frecomposed[:nlevels]), axis=1)

    """locally dominant band"""
    loc_dom_band = np.zeros((f2analyse.shape))

    diff_recomposed = frecomposed[:, 0]
    diff_recomposed.shape = (frecomposed[:, 0].size, 1)
    diff_recomposed = np.append(diff_recomposed, np.diff(frecomposed, axis=1), axis=1).T

    idx = np.argmin(diff_recomposed, axis=0)
    loc_dom_band[f2analyse <= 0] = idx[(f2analyse <= 0)[0]] * (-1)
    idx = np.argmax(diff_recomposed, axis=0)
    loc_dom_band[f2analyse > 0] = idx[(f2analyse > 0)[0]]
    return loc_dom_band, frecomposed
