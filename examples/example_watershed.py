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
import slam.io as sio
import slam.watershed as swat
import slam.sulcal_graph as ssg

###############################################################################
# loading an examplar mesh and corresponding texture
path_to_mesh = "../examples/data/example_mesh.gii"
path_to_mask = None
path_to_output = ""
#path_to_mesh = "/mnt/data/work/BV_database/BV_db_test/subjects/auzias/t1mri/default_acquisition/default_analysis/segmentation/mesh/auzias_Lwhite.gii"
#path_to_output = "/mnt/data/work/BV_database/BV_db_test/subjects/auzias/t1mri/default_acquisition/default_analysis/segmentation/mesh/"

#path_to_mesh = "/mnt/data/work/python_sandBox/brain_slam/debug_watershed/example_mesh.gii"
#path_to_output = "/mnt/data/work/python_sandBox/brain_slam/debug_watershed/"
mesh = sio.load_mesh(path_to_mesh)
side = "left"

###############################################################################
# compute curvature, dpf and voronoi
_, dpf, voronoi = swat.compute_mesh_features(mesh, save=True, outdir=path_to_output, check_if_exist=True)

###############################################################################
# normalize watershed thresholds
thresh_dist, thresh_ridge, thresh_area = swat.normalize_thresholds(voronoi, thresh_dist=20.0, thresh_ridge=1.5,
                                                                   thresh_area=50.0, side=side)
thresh_dist, thresh_ridge, thresh_area = 0,0,0
###############################################################################
# define the exclusion mask (cingular pole)
if path_to_mask is not None:
    mask = sio.load_texture(path_to_mask).darray[0]
else:
    mask = None

###############################################################################
# extract sulcal pits and associated basins
basins, ridges, adjacency = swat.watershed(mesh, voronoi, dpf, thresh_dist, thresh_ridge, thresh_area, mask)

###############################################################################
# generate the textures from watershed outputs
tex_labels, tex_pits, tex_ridges = swat.get_textures_from_dict(mesh, basins, ridges, save=True, outdir=path_to_output)

###############################################################################
# generate the sulcal graph
g = ssg.get_sulcal_graph(adjacency, basins, ridges, save=True, outdir=path_to_output)

###############################################################################
# generate the textures from graph
tex_labels, tex_pits, tex_ridges = ssg.get_textures_from_graph(g, mesh, save=True, outdir=path_to_output)