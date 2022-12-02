"""Spectral Analysis of Gyrification (SPANGY)


This module implements the Spectral Analysis framework described in [1]_ and
applied in [2]_. Hence this module:

1. Perform spectral decomposition of a mesh (eigenvalues and eigenfunctions
   of the laplace-beltrami operator defined on the mesh), i.e. obtaining a
   functional basis
2. Decompose a scalar function living on the mesh (e.g. mean
   curvature) in this functional basis
3. Compute the power spectrum associated to the obtained spectral decomposition
4. Group power spectrum into frequency bands of interest


References
----------
.. [1] D. Germanaud*, J. Lefevre*,R. Toro, C. Fischer, J.Dubois, L.
 Hertz-Pannier, J.F. Mangin, Larger is twistier:Spectral Analysis of
 Gyrification (SPANGY) applied to adult brain size polymorphism,
 Neuroimage, 63 (3), 1257-1272, 2012.

.. [2] D. Germanaud, J. Lefèvre, C. Fischer, M. Bintner, A. Curie, V.
 des Portes, S. Eliez, M. Elmaleh-Bergès, D. Lamblin, S. Passemard,
 G. Operto, M. Schaer, A. Verloes, R. Toro, J.F. Mangin, L. Hertz-Pannier,
 Simplified gyral pattern in severe developmental microcephalies?
 New insights from allometric modeling for spatial and spectral analysis
 of gyrification,NeuroImage,Volume 102, Part 2, 2014,
 Pages 317-331, ISSN 1053-8119,
"""

import numpy as np
from scipy.sparse.linalg import eigsh

import slam.differential_geometry as sdg


def eigenpairs(mesh, nb_eig):
    """Compute nb_eig eigen pairs (eigen value and associated eigenvector) of
    the Laplace-Beltrami operator defined on the input mesh

    The nb_eig smallest eigen pairs of the spectrum of the discrete
    Laplace-Beltrami operator.

    Parameters
    ----------
    mesh : Base.Trimesh

    nb_eig : Int
        number of eigenpairs to compute.
    Returns
    -------
    eig_val : Array of float
        eigenvalues computed.
    eig_vec : Array of float
        eigenvectors computed.
    Notes
    -----
    The nb_eig smallest of the Laplacian spectrum are computed using the
    shift-invert method of eighsh as LAPACK library is far more efficient at
    finding the largest eigenvalues of a matrix
    (see https://docs.scipy.org/doc/scipy/tutorial/arpack.html) for complete
    explanations
    """
    lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='fem')
    eig_val, eig_vec = eigsh(lap.tocsr(), nb_eig, M=lap_b.tocsr(),
                             sigma=1e-6, which='LM')
    return eig_val, eig_vec, lap_b.tocsr()


def spectrum(f2analyse, mass_matrix, eig_vec, eig_val):
    """Compute the (grouped) spectrum of a scalar function  in a functional
    basis

    The functional basis is derived from the eigenvectors of a laplace
    beltrami operator defined on the mesh. The spectrum function:
        1. Compute the full discrete spectrum of the scalar function by
        computing a scalar product between this function and the eigenvectors
        corrected by the mass matrix associated with the mesh
        2. Compute the number of bands based on eigen values
        3. Defined the extents of the different bands
        4. Compute the power associated with each defined spectral band

    Parameters
    ----------
    f2analyse : Array of floats
        function to analyze (mean curvature).
    mass_matrix : Array of floats
        Used in the discretization of the eigenvalue problem.
    eig_vec : Array of floats
        Eigenvectors.
    eig_val : Array of floats
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

    coefficients = f2analyse.dot(mass_matrix.transpose().dot(eig_vec))
    nlevels = int(0.5 * np.log(eig_val[-1] / eig_val[1]) / np.log(2))
    grouped_spectrum = np.zeros((nlevels + 2, 1))
    grouped_spectrum[0] = coefficients[0]**2
    group_indices = np.zeros((nlevels + 2, 2), dtype=int)
    group_indices[0, :] = [0, 0]

    for k in range(nlevels):
        indice = np.where(eig_val >= eig_val[1] * 2 ** (2 * (k)))
        group_indices[k + 1, 0] = indice[0][0]
        indice = np.where(eig_val <= eig_val[1] * 2 ** (2 * (k + 1)))
        group_indices[k + 1, 1] = indice[0][-1]
        grouped_spectrum[k + 1] = \
            np.sum(coefficients[
                   group_indices[k + 1, 0]:group_indices[k + 1, 1] + 1]**2)

    group_indices[-1, 0] = group_indices[-2, 1] + 1
    group_indices[-1, 1] = eig_val.size - 1
    grouped_spectrum[-1] = \
        np.sum(coefficients[
               group_indices[-1, 0]:group_indices[-1, 1]]**2)

    return grouped_spectrum.squeeze(), group_indices, coefficients


def local_dominance_map(
        coefficients, f2analyse, nlevels, group_indices, eig_vec):
    """
    Parameters
    ----------
    coefficients : Array of floats
        Fourier coefficients of the input function f2analyse
    f2analyse : Array of floats
        Scalar function to analyze (e.g. mean curvature)
    nlevels : Array of ints
        number of spectral bands
    group_indices : Array of ints
        indices of spectral bands
    eig_vec : Array of floats
        eigenvectors (reversed order for computation and memory reasons)

    Returns
    -------
    loc_dom_band : Array of floats
        texture with the band contributing the most to f2analyse
    frecomposed : Array of floats
        recomposition of f2analyse in each frequency band
    """
    N = np.size(coefficients)

    frecomposed = np.zeros((len(f2analyse), nlevels - 1), dtype='object')
    eig_vec = np.flip(eig_vec, 1)

    # band by band recomposition
    for i in range(nlevels - 1):
        # levels_ii: number of frequency band wihin the compact Band i
        levels_i = np.arange(
            group_indices[i + 1, 0], group_indices[i + 1, 1] + 1)
        # np.array((number of vertices, number of levels_ii))
        f_ii = np.dot(eig_vec[:, N - levels_i - 1], coefficients[levels_i].T)
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
    idx = idx + 1
    loc_dom_band[f2analyse <= 0] = idx[f2analyse <= 0] * (-1)

    # gyri
    idx = np.argmax(diff_recomposed, axis=1)
    idx = idx + 1
    loc_dom_band[f2analyse > 0] = idx[f2analyse > 0]

    return loc_dom_band, frecomposed
