"""
.. _example_dpf_Boucher:

===================================
Example of depth potential function in slam
===================================
"""


# Authors: Julien Lefevre <julien.lefevre@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# Import of modules
import slam.curvature as sc
import slam.differential_geometry as sdg
import trimesh
import numpy as np
from scipy.spatial import Delaunay

#################################
# Define the example provided in Figure 5 of
# Depth potential function for folding pattern representation,
# registration and analysis
# Maxime Boucher a,b, * , Sue Whitesides a , Alan Evans


def boucher_surface(params, ax, ay, nstep):
    # Parameters
    xmin, xmax = [-ax, ax]
    ymin, ymax = [-ay, ay]
    # Define the sampling
    stepx = (xmax - xmin) / nstep
    stepy = stepx * np.sqrt(3) / 2  # to ensure equilateral faces

    # Coordinates
    x = np.arange(xmin, xmax, stepx)
    y = np.arange(ymin, ymax, stepy)
    X, Y = np.meshgrid(x, y)

    X[::2] += stepx / 2
    X = X.flatten()
    Y = Y.flatten()

    # Delaunay
    faces_tri = Delaunay(np.vstack((X, Y)).T, qhull_options="QJ Qt Qbb")

    # Equation for Z
    M = params[0]
    sigma = params[1]  # called sigma_y in the paper
    Z = (
        M
        / sigma
        * np.exp(-(X**2) - Y**2 / (2 * sigma**2))
        * (Y**2 - sigma**2)
    )

    # Mesh
    coords = np.array([X, Y, Z]).transpose()
    mesh = trimesh.Trimesh(
        faces=faces_tri.simplices,
        vertices=coords,
        process=False)
    return mesh


params = [4, 0.25]
ax = 2
ay = 1
nstep = 50
mesh = boucher_surface(params, ax, ay, nstep)

##################################
# Compute dpf for various alpha
res = sc.curvatures_and_derivatives(mesh)
mean_curvature = res[0].sum(axis=0)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
dpfs = sdg.depth_potential_function(
    mesh, curvature=mean_curvature, alphas=alphas)

amplitude_center = []
amplitude_peak = []
index_peak_pos = np.argmax(mesh.vertices[:, 2])
index_peak_neg = np.argmin(mesh.vertices[:, 2])
for i in range(len(dpfs)):
    amplitude_center.append(dpfs[i][index_peak_neg])
    amplitude_peak.append(dpfs[i][index_peak_pos])

####################################
# Fix alpha and vary M = params[0]
all_M = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
all_amplitudes = []

for M in all_M:
    mesh = boucher_surface([M, 0.25], ax, ay, nstep)
    res = sc.curvatures_and_derivatives(mesh)
    mean_curvature = res[0].sum(axis=0)
    dpfs = sdg.depth_potential_function(
        mesh, curvature=mean_curvature, alphas=[0.0015])
    all_amplitudes.append(dpfs[0][len(mesh.vertices) // 2])

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# import visbrain # visu using visbrain
# import slam.plot as splt
# import matplotlib.pyplot as plt
##########################################################################
# # Visualization of the mesh
# visb_sc = splt.visbrain_plot(
#     mesh=mesh,
#     caption="Boucher mesh",
#     bgcolor=[
#         0.3,
#         0.5,
#         0.7])
# visb_sc
# visb_sc.preview()
###############################################################################
# plt.figure()
# plt.semilogx(alphas, amplitude_center)
# plt.semilogx(alphas, amplitude_peak)
# plt.semilogx(alphas, len(alphas) *
#              [params[0] * (1 + 2 * np.exp(-3 / 2))], "--")
# plt.xlabel("alpha")
# plt.ylabel("amplitude")
# plt.legend(["DPF at center", "DPF (secondary peaks)", "True amplitude"])
# plt.show()
#
######################################
# #  Display dpfs on the surfaces
#
# visb_sc = splt.visbrain_plot(
#     mesh=mesh, tex=dpfs[0], caption="Boucher mesh", bgcolor="white"
# )
# visb_sc = splt.visbrain_plot(
#     mesh=mesh, tex=dpfs[5], caption="Boucher mesh", visb_sc=visb_sc
# )
# visb_sc.preview()
#
##############################################################################
# plt.figure()
# plt.plot(all_M, all_amplitudes, "+-")
# plt.xlabel("M")
# plt.ylabel("Amplitude of DPF")
# plt.show()
