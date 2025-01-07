import numpy as np
from slam import io
from slam import differential_geometry
from slam import texture
from slam import plot
from sandbox.tools import utils
import matplotlib.pyplot as plt


# All objects are GIFTI/NIFTI loaded with slam's io module

def eval_dpf(dpf1, dpf2, mesh=None, display=False):
    """
    Function that computes the difference between 2 dpf from the same mesh
    Returns 
    -------
    object
    """
    diff = dpf1.darray[0] - dpf2.darray[0]
    diff_tex = texture.TextureND(diff)
    # io.write_texture(diff_tex, "~/Documents/UKBIOBANK/utils/diff_tex_dpf.gii")

    if display and mesh is not None:
        visb_sc = plot.visbrain_plot(
            mesh=mesh,
            tex=diff_tex.darray[0],
            caption="DPF",
            cblabel="DPF"
        )
        visb_sc.preview()

    return diff


def eval_labels(labels1, labels2, mesh=None, display=False):
    """
    Function that compute the difference between 2 labels (computed using the watershed for the sulcal pits extraction)
    Returns
    -------
    object
    """
    diff = labels1.darray[0] - labels2.darray[0]
    diff_tex = texture.TextureND(diff)
    # io.write_texture(diff_tex, "~/Documents/UKBIOBANK/utils/diff_tex_dpf.gii")

    if display and mesh is not None:
        visb_sc = plot.visbrain_plot(
            mesh=mesh,
            tex=diff_tex.darray[0],
            caption="Labels",
            cblabel="Labels"
        )
        visb_sc.preview()
    return diff


def eval_mesh_fielder_length(mesh):
    """
    Function that compute the mesh Fielder length using 2 distance type: geodesic and euclidian
    Returns
    -------
    object
    """
    (shortest_path, field_tex_geo) = differential_geometry.mesh_fiedler_length(mesh, dist_type="geodesic")
    fielder_length_geodesic = utils.compute_dist(mesh, shortest_path)

    (fielder_length_euclidian, field_tex_euc) = differential_geometry.mesh_fiedler_length(mesh, dist_type="euclidian")

    print("Min dist with geodesic: ", fielder_length_geodesic)
    print("Min dist with euclidian: ", fielder_length_euclidian)
