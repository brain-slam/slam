import shutil
import os
from sandbox.sample import extract_sulcal_pits


def get_folders(path):
    folders_map = {}
    for folder in os.listdir(path):
        full_path = os.path.join(path, folder)
        if os.path.isdir(full_path):
            folder_id = folder.split("_")[-1]
            folders_map[folder_id] = full_path
    return folders_map


def get_mesh_n_mask(folders):
    meshs = dict()
    masks = dict()
    mesh_path = "t1mri/freesurfer/default_analysis/segmentation/mesh"
    mask_path = "t1mri/freesurfer/default_analysis/segmentation/mesh/surface_analysis"
    for id, dir_path in folders.items():
        meshs[id] = dict()
        masks[id] = dict()
        full_path_mesh = os.path.join(dir_path, mesh_path)
        full_path_mask = os.path.join(dir_path, mask_path)
        # Loop for the mesh
        for file in os.listdir(full_path_mesh):
            if file == "brainmorph_{}_Lwhite.gii".format(id) or file == "brainmorph_{}_Rwhite.gii".format(id):
                splitted_file = file.split("_")
                id = splitted_file[1]
                side = splitted_file[-1][0]
                meshs[id][side] = os.path.join(full_path_mesh, file)
        # Loop for the mask
        for file in os.listdir(full_path_mask):
            if file == "brainmorph_{}_Lwhite_pole_cingular.gii".format(id) or file == "brainmorph_{}_Rwhite_pole_cingular.gii".format(id):
                splitted_file = file.split("_")
                side = splitted_file[2][0]
                masks[id][side] = os.path.join(full_path_mask, file)

    return meshs, masks


def compute_all(meshs, masks, dst):
    map_side = {"L": "left", "R": "right"}
    for id in meshs:
        if not os.path.isdir(os.path.join(dst, id)):
            os.mkdir(os.path.join(dst, id))
        for side in meshs[id]:
            dst_path = os.path.join(dst, id, map_side[side])
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            extract_sulcal_pits(meshs[id][side], dst=dst_path, side=map_side[side], mask_path=masks[id][side])
            shutil.copy(meshs[id][side], os.path.join(dst_path, "mesh.gii"))
        print("Fin {}".format(id))


def compute_one(meshs, masks, dst, id):
    map_side = {"L": "left", "R": "right"}
    mesh = meshs[id]
    mask = masks[id]

    if not os.path.isdir(os.path.join(dst, id)):
        os.mkdir(os.path.join(dst, id))

    for side in mesh:
        dst_path = os.path.join(dst, id, map_side[side])
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        extract_sulcal_pits(mesh[side], dst=dst_path, side=map_side[side], mask_path=mask[side])
        shutil.copy(mesh[side], os.path.join(dst_path, "mesh.gii"))
        print("Fin {}".format(side))


if __name__ == "__main__":
    base_path = "/home/INT/leroux.b/Documents/centreIRMf/"
    base_path_input = os.path.join(base_path, "data")
    # base_path_output = os.path.join(base_path, "sulcals")
    base_path_output = os.path.join(base_path, "new_watershed")

    folders = get_folders(base_path_input)

    meshs, masks = get_mesh_n_mask(folders)
    # compute_all(meshs, masks, dst=base_path_output)
    compute_one(meshs, masks, dst=base_path_output, id="08")
