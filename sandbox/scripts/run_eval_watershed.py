import os
import sys
import numpy as np
sys.path.append("/home/INT/leroux.b/Documents/python_code/slam")
import sandbox.tools.eval_watershed as eval_watershed
import plotly.express as px
from slam import io


def run_eval_dpf(path1, path2, mesh_path=None, display=False):
    dpf1 = io.load_texture(path1)
    dpf2 = io.load_texture(path2)

    mesh = None if mesh_path is None else io.load_mesh(mesh_path)

    diff = eval_watershed.eval_dpf(dpf1, dpf2, mesh, display)
    fig = px.histogram(diff, nbins=100)
    fig.show()


def run_eval_mesh_fielder_length(mesh_path):
    mesh = io.load_mesh(mesh_path)
    eval_watershed.eval_mesh_fielder_length(mesh)


def run_eval_labels(path1, path2, mesh_path=None, display=False):
    labels1 = io.load_texture(path1)
    labels2 = io.load_texture(path2)

    mesh = None if mesh_path is None else io.load_mesh(mesh_path)

    diff = eval_watershed.eval_labels(labels1, labels2, mesh, display)
    fig = px.histogram(diff, nbins=100)
    fig.show()


def main():
    try:
        mode = sys.argv[1].lower()

        path1 = sys.argv[2]
        path2 = sys.argv[3]
    except IndexError:
        print("Error: missing arguments")
        sys.exit()
    try:
        mesh_path = sys.argv[4]
    except IndexError:
        mesh_path = None

    display = True if mesh_path is not None else False

    if mode == "dpf":
        run_eval_dpf(path1, path2, mesh_path=mesh_path, display=display)
    elif mode == "basins":
        run_eval_labels(path1, path2, mesh_path=mesh_path, display=display)
    else:
        print("No execution for the mode:", mode)
        sys.exit()


if __name__ == "__main__":
    main()

    # side_map = {"right": "R", "left": "L"}
    #
    # base_path_bv = "~/Documents/centreIRMf/data"
    # base_path_slam = "~/Documents/centreIRMf/sulcals"
    #
    # id_folder = "08"
    # side = "right"
    # file_side = f"brainmorph_{id_folder}_{side_map[side]}white_basins.gii"
    #
    # mesh_path = os.path.join(base_path_slam, id_folder, side, "mesh.gii")
    #
    # labels_1_path = os.path.join(base_path_slam, id_folder, side, "labels.gii")  # computed with slam
    # labels_2_path = os.path.join(base_path_slam, id_folder, side, file_side)  # computed with BV
    #
    # run_eval_labels(labels_1_path, labels_2_path)

    # mesh_path = os.path.join(base_path, "01/mesh.gii")
    #
    # dpf1_path = os.path.join(base_path, "bv/01/brainmorph_01_Lwhite_DPF.gii")
    # dpf2_path = os.path.join(base_path, "01/dpf.gii")
    #
    # labels1_path = os.path.join(base_path, "bv/01/brainmorph_01_Lwhite_basins.gii")
    #
    # labels2_path = os.path.join(base_path, "01/labels.gii")

    # run_eval_labels(labels1_path, labels2_path, mesh_path, display=True)
    # run_eval_dpf(dpf1_path, dpf2_path, mesh_path, display=False)
    # run_eval_mesh_fielder_length(mesh_path)





