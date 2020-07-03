"""
.. _example_mapping:

===================================
Mapping example in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Importation of slam modules
import slam.generate_parametric_surfaces as sps
import numpy as np
import slam.topology as stop
import slam.plot as splt
import slam.mapping as smap
import slam.distortion as sdst
from vispy.scene import Line
from visbrain.objects import VispyObj, SourceObj

###############################################################################
# Generation of an open mesh
K = [-1, -1]
open_mesh = sps.generate_quadric(K, nstep=[5, 5])
open_mesh_boundary = stop.mesh_boundary(open_mesh)
# Visualization
visb_sc = splt.visbrain_plot(mesh=open_mesh, caption='open mesh')
for bound in open_mesh_boundary:
    points = open_mesh.vertices[bound]
    s_rad = SourceObj('rad', points, color='red', symbol='square',
                      radius_min=10)
    visb_sc.add_to_subplot(s_rad)
    lines = Line(pos=open_mesh.vertices[bound], width=10, color='b')
    # wrap the vispy object using visbrain
    l_obj = VispyObj('line', lines)
    visb_sc.add_to_subplot(l_obj)
visb_sc.preview()

###############################################################################
# Mapping onto a planar disk
disk_mesh = smap.disk_conformal_mapping(open_mesh)
# Visualization
visb_sc2 = splt.visbrain_plot(mesh=disk_mesh, caption='disk mesh')
for bound in open_mesh_boundary:
    points = disk_mesh.vertices[bound]
    s_rad = SourceObj('rad', points, color='red', symbol='square',
                      radius_min=10)
    visb_sc2.add_to_subplot(s_rad)
    lines = Line(pos=disk_mesh.vertices[bound], width=10, color='b')
    # wrap the vispy object using visbrain
    l_obj = VispyObj('line', lines)
    visb_sc2.add_to_subplot(l_obj)
visb_sc2.preview()

###############################################################################
# Compute distortion measures between original and planar representations
angle_diff = sdst.angle_difference(disk_mesh, open_mesh)
area_diff = sdst.area_difference(disk_mesh, open_mesh)
edge_diff = sdst.edge_length_difference(disk_mesh, open_mesh)
print(np.mean(angle_diff))
print(np.mean(area_diff))
print(np.mean(edge_diff))
