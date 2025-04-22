###############################################################################
# importation of slam modules
import slam.io as sio
import slam.watershed as swat
import slam.sulcal_graph as ssg

###############################################################################
# loading an examplar mesh and corresponding texture
path_to_mesh = "../examples/data/example_mesh.gii"
path_to_mask = ""
path_to_output = ""

mesh = sio.load_mesh(path_to_mesh)
side = "left"

###############################################################################
# compute curvature, dpf and voronoi
_, dpf, voronoi = swat.compute_mesh_features(mesh, save=False, outdir=path_to_output, check_if_exist=True)

###############################################################################
# normalize watershed thresholds
thresh_dist, thresh_ridge, thresh_area = swat.normalize_thresholds(mesh, voronoi,
                                                                   thresh_dist=20.0,
                                                                   thresh_ridge=1.5,
                                                                   thresh_area=50.0,
                                                                   side=side)

###############################################################################
# define the exclusion mask (cingular pole)
if path_to_mask:
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