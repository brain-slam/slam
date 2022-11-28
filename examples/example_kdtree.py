"""
.. _example_kdtree:

===================================
KDTree in slam
===================================
"""

# Authors:
# Guillaume Auzias <guillaume.auzias@univ-amu.fr>
# Julien Barrès <julien.barres@etu.univ-amu.fr>

# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2


###############################################################################
# Importation of slam modules
import slam.io as sio

mesh = sio.load_mesh("../examples/data/example_mesh.gii")

###############################################################################
# kdtree serves to compute distances to mesh vertices efficiently
# here we compute the distance between a vector of two points and the mesh
distance, index = mesh.kdtree.query([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
distance
index
