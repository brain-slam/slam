import numpy as np
from slam import io
from slam import differential_geometry
from slam import texture
from sandbox.tools import utils
import matplotlib.pyplot as plt


# All objects are GIFTI/NIFTI loaded with slam's texture io

def eval_dpf(dpf1, dpf2):
    """
    Function that computes the difference between 2 dpf from the same mesh
    Returns
    -------
    object
    """
    diff = dpf1.darray[0] - dpf2.darray[0]
    diff_tex = texture.TextureND(diff)
    io.write_texture(diff_tex, "~/Documents/utils/diff_tex_dpf.gii")
    return diff


def eval_mesh_fielder_length(mesh):
    """
    Function that compute the mesh Fielder length using 2 methods
    Returns
    -------
    object
    """
    (mesh_fiedler_length, field_tex) = differential_geometry.mesh_fiedler_length(mesh, dist_type="geodesic")
    min_mesh_fiedler_length = utils.compute_dist(mesh, mesh_fiedler_length)
    print(min_mesh_fiedler_length)
    (mesh_fiedler_length, field_tex) = differential_geometry.mesh_fiedler_length(mesh, dist_type="euclidian")
    print(mesh_fiedler_length)


def eval_labels(labels1, labels2):
    diff = labels1.darray[0] - labels2.darray[0]
    diff_tex = texture.TextureND(diff)
    io.write_texture(diff_tex, "~/Documents/utils/diff_tex_dpf.gii")
    return diff