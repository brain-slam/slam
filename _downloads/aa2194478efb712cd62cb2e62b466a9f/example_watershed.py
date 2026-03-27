"""
.. _example_watershed:

===================================
Example of watershed in slam
===================================
"""

# Authors:
# Lucile Hashimoto lucile-hashimoto
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# importation of slam modules
import os
import numpy as np
import slam.texture as stex
import slam.io as sio
import slam.watershed as swat

###############################################################################
# loading an examplar mesh and corresponding texture
path_to_mesh = "../examples/data/example_mesh.gii"
path_to_mask = None
path_to_output = None
mesh = sio.load_mesh(path_to_mesh)

###############################################################################
# set the threshold values for the watershed
thresh_dist_perc, thresh_ridge, thresh_area_perc = 10, 0.0001, 3

###############################################################################
# compute curvature, dpf and voronoi
mean_curvature, dpf_star, voronoi = swat.compute_mesh_features(mesh)

###############################################################################
# define the exclusion mask (cingular pole)
if path_to_mask is not None:
    mask = sio.load_texture(path_to_mask).darray[0]
else:
    mask = None


###############################################################################
# extract sulcal pits and associated basins
basins, ridges, adjacency = swat.watershed(
    mesh, voronoi, dpf_star, thresh_dist_perc, thresh_ridge, thresh_area_perc, mask)

###############################################################################
# print basic info
print("nb basins:", len(basins))
print("nb ridges:", len(ridges))

###############################################################################
# print statistics regarding the area of basins
bas_area = swat.get_basins_attribute(basins, attribute='basin_area')
print('average basins area:', np.mean(bas_area))
print('range of basins area:', min(bas_area), '-', max(bas_area))

###############################################################################
# print statistics regarding the height of ridges
rid_height = swat.get_ridges_attribute(ridges, attribute='ridge_depth_diff_min')
print('average ridges height:', np.mean(rid_height))
print('range of ridges height:', min(rid_height), '-', max(rid_height))

###############################################################################
# generate the textures from watershed outputs
atex_labels, atex_pits, atex_ridges = swat.get_textures_from_dict(
    mesh, basins, ridges)

###############################################################################
# print the labels present in the texture of basin labels
print(np.unique(atex_labels))

###############################################################################
# generate the texture of the boundaries between basins from watershed outputs
atex_boundaries = swat.get_texture_boundaries_from_dict(mesh, ridges)

if path_to_output is not None:
    # texture of curvature
    curv_tex = stex.TextureND(darray=mean_curvature)
    sio.write_texture(curv_tex, os.path.join(path_to_output, "mean_curvature.gii"))
    # texture of depth
    dpf_tex = stex.TextureND(darray=dpf_star)
    sio.write_texture(dpf_tex, os.path.join(path_to_output, "dpf_star.gii"))
    # texture of voronoi
    voronoi_tex = stex.TextureND(darray=voronoi)
    sio.write_texture(voronoi_tex, os.path.join(path_to_output, "voronoi.gii"))
    # texture of labels
    tex_labels = stex.TextureND(darray=atex_labels)
    sio.write_texture(tex_labels, os.path.join(path_to_output, "basins.gii"))
    # texture of pits
    tex_pits = stex.TextureND(darray=atex_pits)
    sio.write_texture(tex_pits, os.path.join(path_to_output, "pits.gii"))
    # texture of ridges
    tex_ridges = stex.TextureND(darray=atex_ridges)
    sio.write_texture(tex_ridges, os.path.join(path_to_output, "ridges.gii"))
    # texture of ridges vertices
    texture_boundaries = stex.TextureND(darray=atex_boundaries)
    sio.write_texture(tex_labels, os.path.join(path_to_output, "bondaries.gii"))

#############################################################################
# VISUALIZATION USING plotly
#############################################################################

import slam.plot as splt
# set the parameter to shift the pits and ridges outwards for better visualization
out_shift=0.2
# create a texture combining the DPF* with basins boundaries
tex_plot = dpf_star.copy()
min_dpf = min(dpf_star)
tex_plot[atex_boundaries==1] = min_dpf

display_settings = {}
display_settings['colorbar_label'] = 'DPF*'
mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
intensity_data = {}
intensity_data['values'] = tex_plot
intensity_data["mode"] = "vertex"
fig = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
# add the ridges to the plot
ridges_coords = splt.coords_outward_shift(atex_ridges==1, mesh, out_shift=0.5)
trace_ridges = splt.plot_points(
    ridges_coords,
    marker={"size": 6, "color": "white", "line":{"color":"black", "width":2}},
)
fig.add_trace(trace_ridges)
# add the pits to the plot
pits_coords = splt.coords_outward_shift(atex_pits==1, mesh, out_shift=1)
trace_pits = splt.plot_points(
    pits_coords,
    marker={"size": 6, "color": "red", "line":{"color":"black", "width":2}}
)
fig.add_trace(trace_pits)
fig.show()
fig

# create a texture showing the basins label and boundaries
tex_plot = atex_labels.copy()
tex_plot[atex_boundaries==1] = -1
print(np.unique(tex_plot))

display_settings = {}
display_settings['colorbar_label'] = 'Basins label'
display_settings["colorscale"]= "BrBg"
intensity_data = {}
intensity_data['values'] = tex_plot
intensity_data["mode"] = "vertex"
fig2 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
# add the ridges to the plot
fig2.add_trace(trace_ridges)
# add the pits to the plot
fig2.add_trace(trace_pits)
fig2.show()
fig2

###############################################################################
# extract corresponding sulcal graph
import slam.sulcal_graph as ssg
g = ssg.get_sulcal_graph(adjacency, basins, ridges)
g = ssg.add_coords_attribute(g, mesh,
                                    attribute_vert_index='pit_index',
                                    new_attribute_key='3dcoords')

mesh_data["opacity"]=0.5
fig3 = splt.plot_mesh(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
fig3 = splt.plot_graph_on_mesh(g, mesh, vertex_index_attribute='pit_index', out_shift=1, show_edge_ridge=True, fig=fig3)
# add the ridges to the plot
fig3.add_trace(trace_ridges)
fig3.show()
fig3
