"""
Module for the Spangy class, implementing functions for the spangy package
"""

import slam.io as sio
import numpy as np
import slam.differential_geometry as sdg
import slam.curvature as scurv
from scipy.sparse.linalg import eigsh

def spangy_eigenpairs(mesh,nb_eig):
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
    eigVal, eigVects = eigsh(lap.tocsr(), nb_eig, M=lap_b.tocsr(), sigma=1e-6, which='LM')
    return eigVal, eigVects, lap_b.tocsr()



def spangy_spectrum(f2analyse,MassMatrix,eigVec,eValues):
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
    coefficients = f2analyse.dot(MassMatrix.transpose().dot(eigVec))

    nlevels =int(0.5*np.log(eValues[-1]/eValues[1])/np.log(2))
    grouped_spectrum=np.zeros((nlevels+2,1));
    grouped_spectrum[0] = coefficients[0]**2;
    group_indices = np.zeros((nlevels+2,2),dtype = int)
    group_indices[0,:] = [0, 0]
    
    for k in range(nlevels):
        indice = np.where(eValues>=eValues[1]*2**(2*(k)))
        group_indices[k+1,0] = indice[0][0]
        indice = np.where(eValues<=eValues[1]*2**(2*(k+1)))
        group_indices[k+1,1] = indice[0][-1]
        grouped_spectrum[k+1] = np.sum(coefficients[group_indices[k+1,0]:group_indices[k+1,1]+1]**2)
    
    group_indices[-1,0] = group_indices[-2,1]+1
    group_indices[-1,1] = eValues.size-1
    grouped_spectrum[-1] = np.sum(coefficients[group_indices[-1,0]:group_indices[-1,1]]**2)
    
    return grouped_spectrum,group_indices,coefficients
