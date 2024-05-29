import numpy as np
import sandbox.tools.eval_watershed as eval_watershed
import plotly.express as px
from slam import plot
from slam import io


def run_eval_dpf(path1, path2):
    dpf1 = io.load_texture(dpf1_path)
    dpf2 = io.load_texture(dpf2_path)

    diff = eval_watershed.eval_dpf(dpf1, dpf2)
    fig = px.histogram(diff, nbins=100)
    fig.show()


def run_eval_mesh_fielder_length(mesh_path):
    mesh = io.load_mesh(mesh_path)
    eval_watershed.eval_mesh_fielder_length(mesh)


def run_eval_labels(labels1_path, labels2_path):
    labels1 = io.load_texture(labels1_path)
    labels2 = io.load_texture(labels2_path)

    eval_watershed.eval_labels(labels1, labels2)


if __name__ == "__main__":
    mesh_path = "~/Documents/test_sulcal/01/mesh.gii"

    dpf1_path = "~/Documents/test_sulcal/bv/01/brainmorph_01_Lwhite_DPF.gii"
    dpf2_path = "~/Documents/test_sulcal/01/dpf.gii"

    labels1_path = "~/Documents/test_sulcal/bv/01/brainmorph_01_Lwhite_basins.gii"
    labels2_path = "~/Documents/test_sulcal/01/labels.gii"

    run_eval_labels(labels1_path, labels2_path)

    # run_eval_dpf(dpf1_path, dpf2_path)
    # run_eval_mesh_fielder_length(mesh_path)




