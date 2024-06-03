import os
import shutil


def get_bv_basins(path):
    path_basins = "t1mri/freesurfer/default_analysis/segmentation/mesh/surface_analysis"
    basins_map = dict()
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folder_id = folder.split("_")[-1]
            basins_map[folder_id] = dict()
            full_path = os.path.join(path, folder, path_basins)
            for file in os.listdir(full_path):
                if "basin" in file:
                    if len(file.split(".")) == 2:  # format [filename, extension]
                        side = file.split("_")[2][0]
                        basins_map[folder_id][side] = os.path.join(full_path, file)
    return basins_map


def move_bv_basins(basins_dict, output_path):
    side_map = {"L": "left", "R": "right"}
    for key, values in basins_dict.items():
        if len(values) != 0:
            for side, value in values.items():
                full_output_path = os.path.join(output_path, key, side_map[side])
                new_name = f"{side}_basins_BV.gii"
                shutil.copy(value, os.path.join(full_output_path, new_name))


base_path = "/home/INT/leroux.b/Documents/centreIRMf/data/"
output_path = "/home/INT/leroux.b/Documents/centreIRMf/sulcals/"
basins = get_bv_basins(base_path)
move_bv_basins(basins, output_path)