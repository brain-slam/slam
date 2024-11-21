import numpy as np
import os
import sys
from slam import io
from slam import differential_geometry
from slam import vertex_voronoi
from slam import geodesics
from slam import texture
from slam import curvature
# from slam import plot
# sys.path.append("/home/INT/leroux.b/Documents/python_code/slam")
sys.path.insert(0, os.path.abspath(os.curdir))
import time
import watershed


def display_watershed(mesh_path, filename):
    """
    Function that display a Texture File

    :args: main_path: str to the directory which contains the mesh and the tex file
    :args: filename: str of the tex file (which is in the main_path)
    """
    mesh = io.load_mesh(mesh_path)
    tex_path = os.path.join(filename)
    tex = io.load_texture(tex_path)

    visb_sc = plot.visbrain_plot(
        mesh=mesh,
        tex=tex.darray[0],
        caption=filename.split(".")[0],
        cblabel=filename.split(".")[0],
    )
    visb_sc.preview()


def extract_sulcal_pits(main_path, dst, side="left", mask_path=None):
    """
    Function that compute the extraction of the sulcal pits.
    Be sure to have at least the 'mesh.gii' file

    :args: main_path: str to the directory containing the mesh and the mask (if specified)
    :args: side: str of the side of the brain (left or right)
    :args: mask_path: str of the path to the mask file. If not, the DPF is used as the mask

    This function saves all the computed file in the main_path directory
    """

    mesh = io.load_mesh(main_path)

    start_time = time.time()

    # Compute the curvature
    print("\n\tComputing the curvature\n")
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = curvature.curvatures_and_derivatives(
        mesh)
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    curv_tex = texture.TextureND(mean_curv)
    curv_tex.z_score_filtering(z_thresh=3)
    io.write_texture(curv_tex, os.path.join(dst, "curv_z_score.gii"))

    mean_curv = io.load_texture(os.path.join(dst, "curv_z_score.gii"))
    mean_curv = mean_curv.darray[0]

    # Compute the DPF and save it
    print("\n\tComputing the DPF\n")
    dpf = differential_geometry.depth_potential_function(mesh, mean_curv, [0.47])
    dpf_tex = texture.TextureND(darray=dpf[0])
    io.write_texture(dpf_tex, os.path.join(dst, "dpf.gii"))

    # Compute Voronoi vertex and save it
    print("\n\tComputing Voronoi's vertex\n")
    vert_voronoi = vertex_voronoi.vertex_voronoi(mesh)
    np.save(os.path.join(dst, "vertex.npy"), vert_voronoi)

    print("\n\tComputing the Fiedler geodesic length and surface area\n")
    mesh_area = np.sum(vert_voronoi)

    # (mesh_fiedler_length, field_tex) = differential_geometry.mesh_fiedler_length(mesh, dist_type="geodesic")
    # min_mesh_fiedler_length = utils.compute_dist(mesh, mesh_fiedler_length)
    fielder = differential_geometry.mesh_laplacian_eigenvectors(mesh, 1)
    imin = fielder.argmin()
    imax = fielder.argmax()
    dist = geodesics.compute_gdist(mesh, imin)
    min_mesh_fiedler_length = dist[imax]

    thresh_dist = 20
    thresh_ridge = 1.5
    thresh_area = 50
    group_average_Fiedler_length = 238.25 if side == "right" else 235.95
    group_average_surface_area = 91433.68 if side == "right" else 91369.33

    thresh_dist *= min_mesh_fiedler_length / group_average_Fiedler_length
    thresh_area *= mesh_area / group_average_surface_area

    if mask_path is not None:
        maskTex = io.load_texture(mask_path)
        mask = np.array(maskTex.darray[0])
    else:
        mask = np.zeros(dpf[0].shape)

    print("\n\tComputing the watershed\n")
    labels_1, pitsKept_1, pitsRemoved_1, ridgePoints, parent_1 = watershed.watershed(
        mesh,
        vert_voronoi,
        dpf_tex.darray,
        mask,
        thresh_dist,
        thresh_ridge
    )

    labels, infoBasins, pitsKept, pitsRemoved_2, parent = watershed.areaFiltering(mesh, vert_voronoi, labels_1,
                                                                                  pitsKept_1, parent_1, thresh_area)
    pitsRemoved = pitsRemoved_1 + pitsRemoved_2

    end_time = time.time()
    print("Time elapsed: {:.5f} secondes".format(end_time - start_time))

    io.write_texture(texture.TextureND(darray=labels.flatten()), os.path.join(dst, "labels.gii"))

    # texture of pits
    atex_pits = np.zeros((len(labels), 1))
    for pit in pitsKept:
        atex_pits[int(pit[0])] = 1
    io.write_texture(texture.TextureND(darray=atex_pits.flatten()), os.path.join(dst, "pits_tex.gii"))

    # texture of noisy pits
    atex_noisypits = np.zeros((len(labels), 1))
    for pit in pitsRemoved:
        atex_noisypits[int(pit[0])] = 1
    io.write_texture(texture.TextureND(darray=atex_noisypits.flatten()), os.path.join(dst, "noisy_pits_tex.gii"))

    # texture of ridges
    atex_ridges = np.zeros((len(labels), 1))
    for ridge in ridgePoints:
        atex_ridges[ridge[2]] = 1
    io.write_texture(texture.TextureND(darray=atex_ridges.flatten()), os.path.join(dst, "rigdes_tex.gii"))


if __name__ == "__main__":
    try:
        run = sys.argv[1]
        path = sys.argv[2]
    except IndexError:
        print("Starting error")
        print("You have to run: ")
        print("python sample.py exec PATH_TO_MESH SIDE_H MASK_FILENAME")
        print("or")
        print("python sample.py display PATH_TO_MESH FILENAME")
        sys.exit()

    if run == "exec":
        side = sys.argv[3].lower()
        try:
            mask = sys.argv[4]
        except IndexError:
            mask = None

        extract_sulcal_pits(path, side, mask)

    elif run == "display":
        filename = sys.argv[3]

        display_watershed(path, filename)

    else:
        print("No option for this run mode")
