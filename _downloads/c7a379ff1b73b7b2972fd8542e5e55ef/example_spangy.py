"""
.. _example_spangy:

===================================
example of SPANGY (spectral decomposition) tools in slam
===================================
"""

# Authors:

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# importation of slam modules
import numpy as np
import slam.io as sio
import slam.curvature as scurv
import slam.spangy as spgy

###############################################################################
# LOAD MESH
mesh = sio.load_mesh(
    '../examples/data/example_mesh.gii')
vertices = mesh.vertices
num_vertices = len(vertices)
print('{} vertices'.format(num_vertices))

N = 1500  # N should be < to the number of vertices.

###############################################################################
# Compute eigenpairs and mass matrix
eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

###############################################################################
# CURVATURE
PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
    scurv.curvatures_and_derivatives(mesh)
mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

###############################################################################
# WHOLE BRAIN MEAN-CURVATURE SPECTRUM
grouped_spectrum, group_indices, coefficients, nlevels \
    = spgy.spectrum(mean_curv, lap_b, eigVects, eigVal)
levels = len(group_indices)

# a. Whole brain parameters
mL_in_MM3 = 1000
CM2_in_MM2 = 100
volume = mesh.volume
surface_area = mesh.area
afp = np.sum(grouped_spectrum[1:])
print('** a. Whole brain parameters **')
print('Volume = %d mL, Area = %d cm², Analyze Folding Power = %f,' %
      (np.floor(volume / mL_in_MM3), np.floor(surface_area / CM2_in_MM2), afp))

# b. Band number of parcels
print('** b. Band number of parcels **')
print('B4 = %f, B5 = %f, B6 = %f' % (0, 0, 0))

# c. Band power
print('** c. Band power **')
print('B4 = %f, B5 = %f, B6 = %f' %
      (grouped_spectrum[4], grouped_spectrum[5],
       grouped_spectrum[6]))

# d. Band relative power
print('** d. Band relative power **')
print('B4 = %0.5f, B5 = %0.5f , B6 = %0.5f' %
      (grouped_spectrum[4] / afp, grouped_spectrum[5] / afp,
       grouped_spectrum[6] / afp))

###############################################################################
# LOCAL SPECTRAL BANDS
loc_dom_band, frecomposed = spgy.local_dominance_map(coefficients, mean_curv,
                                                     levels, group_indices,
                                                     eigVects)


# WHOLE BRAIN MEAN-CURVATURE<=0 & MEAN-CURVATURE>0 SPECTRI
# --------------------------------------------------------
# Define negative mean curvature subsignal
mean_curv_sulci = np.zeros((mean_curv.shape))
mean_curv_sulci[mean_curv <= 0] = mean_curv[mean_curv <= 0]
grouped_spectrum_sulci, group_indices_sulci, coefficients_sulci, _ \
    = spgy.spectrum(mean_curv_sulci, lap_b, eigVects, eigVal)

# Define positive mean curvature subsignal
mean_curv_gyri = np.zeros((mean_curv.shape))
mean_curv_gyri[mean_curv > 0] = mean_curv[mean_curv > 0]
grouped_spectrum_gyri, group_indices_gyri, coefficients_gyri, _ \
    = spgy.spectrum(mean_curv_gyri, lap_b, eigVects, eigVal)

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# import slam.plot as splt
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import pyvista as pv
#
# ###############################################################################
# # Plot of mean curvature on the mesh
# visb_sc = splt.visbrain_plot(
#     mesh=mesh,
#     tex=mean_curv,
#     caption='Mean Curvature',
#     cmap='jet')
# visb_sc.preview()
#
# # Plot coefficients and bands for all mean curvature signal
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.scatter(np.sqrt(eigVal/2*np.pi),
# coefficients, marker='+', s=10, linewidths=0.5)
# #ax1.plot(np.sqrt(eigVal[1:]) / (2 * np.pi),
# # coefficients[1:]) # remove B0 coefficients
# #ax1.scatter(np.sqrt(eigVal[1:]/2*np.pi),
# # coefficients[1:], marker='+', s=10, linewidths=0.5) # remove B0 coefficients
# ax1.set_xlabel('Frequency (m⁻¹)')
# ax1.set_ylabel('Coefficients')
#
# # print(grouped_spectrum)
# ax2.bar(np.arange(0, levels), grouped_spectrum)
# #ax2.bar(np.arange(1, nlevels), grouped_spectrum[1:]) # remove B0
# ax2.set_xlabel('Spangy Frequency Bands')
# ax2.set_ylabel('Power Spectrum')
# plt.show()
#
# # Plot of spectral dominant bands on the mesh
# visb_sc = splt.visbrain_plot(mesh=mesh, tex=loc_dom_band,
#                              caption='Local Dominant Band', cmap='jet')
# visb_sc.preview()
#
# # Plot mean curvature coefficients & compacted power spectrum characterizing
# # either Sulci either Gyri folding pattern
# # ---------------------------------------------------------------------------
# coefficients_colors_sulci \
#     = plot_global_coefficients_and_bands_sulci_or_gyri(
#     group_indices_sulci,
#     coefficients_sulci,
#     colormap_sulci
# )
# coefficients_colors_gyri \
#     = plot_global_coefficients_and_bands_sulci_or_gyri(
#     group_indices_gyri,
#     coefficients_gyri,
#     colormap_gyri
# )
#
# fig, axs = plt.subplots(2, 2)
#
# # GLOBAL FOLDING PATTERN OF SULCI
# #axs[0,0].plot(np.sqrt(eigVal/2*np.pi),
# # coefficients_sulci, color=coefficients_colors_sulci)
# axs[0,0].scatter(np.sqrt(eigVal/2*np.pi),
#                  coefficients_sulci, marker='+',
#                  s=10, linewidths=0.5, color=coefficients_colors_sulci)
# #axs[0,0].scatter(np.sqrt(eigVal[1:]/2*np.pi),
# # coefficients_sulci[1:], marker='+', s=10, linewidths=0.5,
# # color=coefficients_colors_sulci[1:]) # remove B0 coefficient
# axs[0,0].set_xlabel('Frequency (m⁻¹)')
# axs[0,0].set_ylabel('Coefficients mean_curv<=0')
#
# axs[0,1].bar(np.arange(0, nlevels),
# grouped_spectrum_sulci.squeeze(), color=colormap_sulci)
# #axs[0,1].bar(np.arange(1, nlevels),
# # grouped_spectrum_sulci[1:].squeeze(),
# # color=colormap_sulci[1:]) # remove B0
# axs[0,1].set_xlabel('Spangy frequency bands')
# axs[0,1].set_ylabel('Power spectrum mean_curv<=0')
#
# # GLOBAL FOLDING PATTERN OF GYRI
# #axs[1,0].plot(np.sqrt(eigVal/2*np.pi),
# # coefficients_gyri, color=coefficients_colors_gyri)
# axs[1,0].scatter(np.sqrt(eigVal/2*np.pi),
#                  coefficients_gyri, marker='+',
#                  s=10, linewidths=0.5, color=coefficients_colors_gyri)
# #axs[1,0].scatter(np.sqrt(eigVal[1:]/2*np.pi),
# # coefficients_gyri[1:], marker='+', s=10, linewidths=0.5,
# # color=coefficients_colors_gyri[1:]) # remove B0 coefficient
# axs[1,0].set_xlabel('Frequency (m⁻¹)')
# axs[1,0].set_ylabel('Coefficients mean_curv>0')
#
# axs[1,1].bar(np.arange(0, nlevels),
#              grouped_spectrum_gyri.squeeze(), color=colormap_gyri)
# #axs[1,1].bar(np.arange(1, nlevels),
# # grouped_spectrum_gyri[1:].squeeze(), color=colormap_gyri[1:]) # remove B0
# axs[1,1].set_xlabel('Spangy frequency bands')
# axs[1,1].set_ylabel('Power spectrum mean_curv>0')
#
# plt.show()
#
# # LOCAL SPECTRAL BANDS
# # --------------------
# # Plot of spectral dominant bands on the mesh, with automatized colormap
# number_of_displayed_bands = len(np.unique(loc_dom_band))
#
# band_values = np.linspace(np.min(loc_dom_band),
#                           np.max(loc_dom_band),
#                           number_of_displayed_bands+1)
# # 13 = 6 positive bands * 2 + Band 0 --> to generalize
# band_colors = np.empty((number_of_displayed_bands+1, 4))
# limit = np.max(loc_dom_band) - number_of_high_folding_pattern_bands + 1
# # limit band number under which band is displayed in black
#
# band_colors[band_values < limit] = colors[0, :] # low frequency bands
#
# # associate one color per high frequency sulcus band
# for i in range(number_of_high_folding_pattern_bands):
#     band_colors[band_values == -(limit+i)] = colors[i+1, :]
#
# # associate one color per high frequency gyrus band
# for i in range(number_of_high_folding_pattern_bands):
#     band_colors[band_values ==
#     (limit+i)] = colors[i+1+number_of_high_folding_pattern_bands, :]
#
# localbands_colormap = ListedColormap(band_colors)
# #loc_dom_band = loc_dom_band.astype(int)
#
# p = pv.Plotter()
# p.add_mesh(mesh, scalars=loc_dom_band,
#            show_edges=False, cmap=localbands_colormap, show_scalar_bar=False)
# p.add_text("Local Dominant Bands", font_size=14)
# p.add_scalar_bar('Band n°', fmt="%.0f")
# p.show()
#
# #################################################
# # DETAILED AND GENERALIZED DISPLAY OPTIONS #
# #################################################
#
# # AUTOMATIZED COLORMAP DEPENDING ON NUMBER OF BANDS
# # -------------------------------------------------
# # Define automatically the sulci/gyri folding pattern colormap,
# # in correlation with nlevels --> colors, colormap_sulci, colormap_gyri
# number_of_high_folding_pattern_bands = int(nlevels/2)
#
# colors = np.zeros((2*number_of_high_folding_pattern_bands+1, 4))
# # 4 colors for 7 bands (black + B4 + B5 + B6)
# color_variations_coef = np.linspace(0, 1, number_of_high_folding_pattern_bands)
#
# # low frequency bands
# colors[0, :] = np.array([0, 0, 0, 1])
# # to display in grey: np.array([0.75, 0.75, 0.75, 1])
#
# # colors of high-folding-pattern sulci bands
# for i in range(number_of_high_folding_pattern_bands):
#     colors[i+1, :] = np.array([0, color_variations_coef[i], 1, 1])
#
# # colors of high-folding-pattern gyri bands
# for i in range(number_of_high_folding_pattern_bands):
#     colors[i+1+number_of_high_folding_pattern_bands, :] \
#         = np.array([1, color_variations_coef[i], 0, 1])
#
# # allocate dedicated color to each bar of the global power spectrum:
# # Colormap for Sulci folding patterns
# colormap_sulci = [] # len = 7
# for i in np.arange(0, nlevels-number_of_high_folding_pattern_bands):
#     colormap_sulci.append(colors[0, :]) # B0, B-1, B-2, B-3
# for i in np.arange(0, number_of_high_folding_pattern_bands):
#     colormap_sulci.append(colors[i+1, :]) # B-4, B-5, B-6
#
# # Colormap for Gyri folding patterns
# colormap_gyri = [] # len = 7
# for i in np.arange(0, nlevels-number_of_high_folding_pattern_bands):
#     colormap_gyri.append(colors[0, :]) # B0, B1, B2, B3
# for i in np.arange(0, number_of_high_folding_pattern_bands):
#     colormap_gyri.append(
#     colors[i+1+number_of_high_folding_pattern_bands, :]) # B4, B5, B6
#
# # GLOBAL COEFFICIENTS AND BANDS
# # -----------------------------
# def plot_global_coefficients_and_bands_sulci_or_gyri(
# group_indices, coefficients, colormap):
#     coefficients_colors = [] # len = N (e.g. 1500 eigenpairs)
#
#     # coefficient corresponding to band B0
#     coefficients_colors.append(colormap[0])
#
#     # coefficients corresponding to other bands i.e. B1..B6 & B-1..B-6
#     band_last_eigenvalue = []
#     for i in range(len(group_indices)):
#         band_last_eigenvalue.append(group_indices[i][1])
#         # i, first eigen value of the band Bi
#
#     for i in np.arange(1, len(coefficients)): # len(coefficients) = N
#         j = 0
#         while i > band_last_eigenvalue[j]:
#             j = j+1
#             if i < band_last_eigenvalue[j]:
#                 break
#
#         coefficients_colors.append(colormap[j])
#
#     return coefficients_colors
