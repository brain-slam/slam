"""
.. _example_remeshing:

===================================
Remeshing example in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: MIT
# sphinx_gallery_thumbnail_number = 2

###############################################################################
# NOTE: there is no visualization tool in slam, but we provide at the
# end of this script exemplare code to do the visualization with
# an external solution
###############################################################################

###############################################################################
# Importation of slam modules
import slam.io as sio
import slam.remeshing as srem

###############################################################################
# Source object files
source_mesh_file = "../examples/data/example_mesh.gii"
source_texture_file = "../examples/data/example_texture.gii"
source_spherical_mesh_file = "../examples/data/example_mesh_spherical.gii"

###############################################################################
# Target object files
target_mesh_file = "../examples/data/example_mesh_2.gii"
target_spherical_mesh_file = "../examples/data/example_mesh_2_spherical.gii"

source_mesh = sio.load_mesh(source_mesh_file)
source_tex = sio.load_texture(source_texture_file)
source_spherical_mesh = sio.load_mesh(source_spherical_mesh_file)

target_mesh = sio.load_mesh(target_mesh_file)
target_spherical_mesh = sio.load_mesh(target_spherical_mesh_file)

interpolated_tex_values = srem.spherical_interpolation_nearest_neigbhor(
    source_spherical_mesh, target_spherical_mesh, source_tex.darray[0]
)

#############################################################################
# VISUALIZATION USING EXTERNAL TOOLS
#############################################################################
# # Visualization with visbrain
# import slam.plot as splt
# ###############################################################################
# visb_sc = splt.visbrain_plot(
#     mesh=source_mesh,
#     tex=source_tex.darray[0],
#     caption="source with curvature",
#     cblabel="curvature",
# )
# visb_sc = splt.visbrain_plot(
#     mesh=source_spherical_mesh,
#     tex=source_tex.darray[0],
#     caption="spherical source mesh",
#     cblabel="curvature",
#     visb_sc=visb_sc,
# )
# visb_sc = splt.visbrain_plot(
#     mesh=target_mesh,
#     tex=interpolated_tex_values,
#     caption="target mesh with curvature " "from source mesh",
#     cblabel="curvature",
#     visb_sc=visb_sc,
# )
# visb_sc.preview()
