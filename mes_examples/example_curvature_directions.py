"""
.. _example_curvature:

===================================
example of curvature estimation in slam
===================================
"""

# Authors: Julien Lefevre <julien.lefevre@univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# importation of slam modules
import slam.utils as ut
import slam.generate_parametric_surfaces as sgps
import slam.plot as splt
import slam.curvature as scurv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###############################################################################
# Visualization of vector fields with matplotlib

def visualize(mesh,vector_field,colors=None,params=None):
    """
    Visualize a mesh and a vector field over it
    :param mesh: a mesh with n points
    :param vector_field: (n,3) array
    :param colors: (n,3) array
    :param params: params[0] is the length of the quivers
    :return:
    """
    n = mesh.vertices.shape[0]
    if colors == None:
        colors = np.zeros((n,3))
        colors[:, 0] = 1
    if params == None:
        params = []
        params.append(0.1)

    fig=plt.figure()
    ax=Axes3D(fig)
    ax.plot_trisurf(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
                    triangles=mesh.faces, shade=True)
    plt.quiver(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:,2],
               vector_field[:, 0], vector_field[:, 1], vector_field[:, 2], length=params[0], colors=colors)

    return fig

###############################################################################
# Create quadric mesh

nstep = 20
equilateral = True

K=[1, 0.5]
quadric_mesh = sgps.generate_quadric(K, nstep=[int(nstep),int(nstep)], equilateral=equilateral, ax=1, ay=1, random_sampling=False,
                     ratio=0.2, random_distribution_type='gaussian')

###############################################################################
# Compute principal directions of curvature

PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(quadric_mesh)

###############################################################################
# Visualization

visualize(quadric_mesh, PrincipalDir1)
plt.show()
