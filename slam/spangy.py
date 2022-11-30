"""
Spectral Analysis of Gyrification
This module implements the Spectral Analysis described in D. Germanaud*, J.
Lefevre*, R. Toro, C. Fischer, J.Dubois, L. Hertz-Pannier, J.F. Mangin,
Larger is twistier: Spectral Analysis of Gyrification (SPANGY) applied
to adult brain size polymorphism,
Neuroimage, 63 (3), 1257-1272, 2012.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
import slam.differential_geometry as sdg


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
    eigVal, eigVects = eigsh(lap.tocsr(), nb_eig, M=lap_b.tocsr(),
                             sigma=1e-6, which='LM')
    return eigVal, eigVects, lap_b.tocsr()


def spectrum(f2analyse, MassMatrix, eigVec, eValues):
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
    coefficients : Array of floats
        Fourier coefficients of the input function f2analyse.
    """

    coefficients = f2analyse.dot(MassMatrix.transpose().dot(eigVec))
    nlevels = int(0.5 * np.log(eValues[-1] / eValues[1]) / np.log(2))
    grouped_spectrum = np.zeros((nlevels + 2, 1))
    grouped_spectrum[0] = coefficients[0]**2
    group_indices = np.zeros((nlevels + 2, 2), dtype=int)
    group_indices[0, :] = [0, 0]

    for k in range(nlevels):
        indice = np.where(eValues >= eValues[1] * 2**(2 * (k)))
        group_indices[k + 1, 0] = indice[0][0]
        indice = np.where(eValues <= eValues[1] * 2**(2 * (k + 1)))
        group_indices[k + 1, 1] = indice[0][-1]
        grouped_spectrum[k +
                         1] = np.sum(coefficients[group_indices[k +
                                                  1, 0]:group_indices[k +
                                                  1, 1] + 1]**2)

    group_indices[-1, 0] = group_indices[-2, 1] + 1
    group_indices[-1, 1] = eValues.size - 1
    grouped_spectrum[-1] = np.sum(coefficients[group_indices[-1,
                                               0]:group_indices[-1,
                                               1]]**2)

    return grouped_spectrum, group_indices, coefficients


def local_dominance_map(
        coefficients, f2analyse, nlevels, group_indices, eigVec):
    """
    Parameters
    ----------
    coefficients : Array of floats
        Fourier coefficients of the input function f2analyse
    f2analyse : Array of floats
        function to analyze (mean curvature)
    nlevels : Array of ints
        number of spectral bands
    group_indices : Array of ints
        indices of spectral bands
    eigVec : Array of floats
        eigenvectors (reversed order for computation and memory reasons)

    OUTPUTS
    loc_dom_band : Array of floats
        texture with the band contributing the most to f2analyse
    frecomposed : Array of floats
        recomposition of f2analyse in each frequency band
    """
    N = np.size(coefficients)

    frecomposed = np.zeros((len(f2analyse), nlevels - 1), dtype='object')
    eigVec = np.flip(eigVec, 1)

    # band by band recomposition
    for i in range(nlevels - 1):
        # levels_ii: number of frequency band wihin the compact Band i
        levels_i = np.arange(
            group_indices[i + 1, 0], group_indices[i + 1, 1] + 1)
        # np.array((number of vertices, number of levels_ii))
        f_ii = np.dot(eigVec[:, N - levels_i - 1], coefficients[levels_i].T)
        frecomposed[:, i] = f_ii

    # locally dominant band
    loc_dom_band = np.zeros((f2analyse.shape))

    diff_recomposed = frecomposed[:, 0]
    diff_recomposed = np.concatenate(
        (np.expand_dims(
            diff_recomposed, axis=1), np.diff(
            frecomposed, axis=1)), axis=1)

    # sulci
    idx = np.argmin(diff_recomposed, axis=1)
    loc_dom_band[f2analyse <= 0] = idx[f2analyse <= 0] * (-1)

    # gyri
    idx = np.argmax(diff_recomposed, axis=1)
    loc_dom_band[f2analyse > 0] = idx[f2analyse > 0]

    return loc_dom_band, frecomposed
