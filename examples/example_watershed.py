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
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# importation of slam modules
import os
import slam.texture as stex
import slam.io as sio
import slam.watershed as swat

###############################################################################
# loading an examplar mesh and corresponding texture
path_to_mesh = "../examples/data/example_mesh.gii"
path_to_mask = None
path_to_output = None
#path_to_mesh = "/mnt/data/work/BV_database/BV_db_test/subjects/auzias/t1mri/default_acquisition/default_analysis/segmentation/mesh/auzias_Lwhite.gii"
#path_to_output = "/mnt/data/work/BV_database/BV_db_test/subjects/auzias/t1mri/default_acquisition/default_analysis/segmentation/mesh/"

#path_to_mesh = "/mnt/data/work/python_sandBox/brain_slam/debug_watershed/example_mesh.gii"
#path_to_output = "/mnt/data/work/python_sandBox/brain_slam/debug_watershed/"
mesh = sio.load_mesh(path_to_mesh)
side = "left"

###############################################################################
# compute curvature, dpf and voronoi
mean_curvature, dpf, voronoi = swat.compute_mesh_features(mesh)

###############################################################################
# normalize watershed thresholds
thresh_dist, thresh_ridge, thresh_area = swat.normalize_thresholds(voronoi, thresh_dist=20.0, thresh_ridge=1.5,
                                                                   thresh_area=50.0, side=side)
thresh_dist, thresh_ridge, thresh_area = 0, 0, 0
###############################################################################
# define the exclusion mask (cingular pole)
if path_to_mask is not None:
    mask = sio.load_texture(path_to_mask).darray[0]
else:
    mask = None

###############################################################################
# extract sulcal pits and associated basins
basins, ridges, adjacency = swat.watershed(
    mesh, voronoi, dpf, thresh_dist, thresh_ridge, thresh_area, mask)

###############################################################################
# generate the textures from watershed outputs
atex_labels, atex_pits, atex_ridges = swat.get_textures_from_dict(
    mesh, basins, ridges)

# generate the texture of the boundaries between basins from watershed outputs
atex_boundaries = swat.get_texture_boundaries_from_dict(mesh, ridges)

if path_to_output is not None:
    # texture of curvature
    curv_tex = stex.TextureND(darray=mean_curvature)
    sio.write_texture(curv_tex, os.path.join(path_to_output, "mean_curvature.gii"))
    # texture of depth
    dpf_tex = stex.TextureND(darray=dpf)
    sio.write_texture(dpf_tex, os.path.join(path_to_output, "dpf.gii"))
    # texture of voronoi
    voronoi_tex = stex.TextureND(darray=voronoi)
    sio.write_texture(voronoi_tex, os.path.join(path_to_output, "voronoi.gii"))
    # texture of labels
    tex_labels = stex.TextureND(darray=atex_labels)
    sio.write_texture(tex_labels, os.path.join(path_to_output, "labels.gii"))
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

# create a texture combining the outputs from the watershed
tex_plot = atex_labels
tex_plot[atex_boundaries==1] = 1
tex_plot[atex_pits==1] = 1

display_settings = {}
display_settings['colorbar_label'] = 'Basins labels'
mesh_data = {}
mesh_data['vertices'] = mesh.vertices
mesh_data['faces'] = mesh.faces
mesh_data['title'] = 'Basins Labels'
intensity_data = {}
intensity_data['values'] = tex_plot
intensity_data["mode"] = "vertex"
Fig = splt.mesh_projection(
    mesh_data=mesh_data,
    intensity_data=intensity_data,
    display_settings=display_settings)
Fig.show()
