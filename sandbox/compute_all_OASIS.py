import shutil
import os
import sys
sys.path.insert(0, os.path.abspath(os.curdir))
from sandbox.sample import extract_sulcal_pits


def get_subj(path):
    subjects_list = list()
    for folder in os.listdir(path):
        if "MR2" in folder:
            if "OAS1" in folder:
                subjects_list.append(folder)
    return subjects_list


def get_mesh(subjs, path):
    meshs_map = dict()

    r_mesh = 'surf/rh.white.gii'
    l_mesh = 'surf/lh.white.gii'
    for subj in subjs:
        r_mesh_path = os.path.join(path, subj, r_mesh)
        l_mesh_path = os.path.join(path, subj, l_mesh)
        meshs_map[subj] = {
            "right": r_mesh_path,
            "left": l_mesh_path
        }
    return meshs_map

def get_mask(subjs, path):
    masks_map = dict()

    for subj in subjs:
        r_mask = "{}_Rwhite_pole_cingular.gii".format(subj)
        l_mask = "{}_Lwhite_pole_cingular.gii".format(subj)

        r_mesh_path = os.path.join(path, subj, r_mask)
        l_mesh_path = os.path.join(path, subj, l_mask)
        masks_map[subj] = {
            "right": r_mesh_path,
            "left": l_mesh_path
        }
    return masks_map


def compute_all(meshs, masks, dst_path, nb_subj=5):
    cpt_subj = 0
    for subj in meshs:
        if not os.path.isdir(os.path.join(dst_path, subj)):
            os.mkdir(os.path.join(dst_path, subj))
        for side in meshs[subj]:
            dst_full_path = os.path.join(dst_path, subj, side)
            if not os.path.isdir(dst_full_path):
                os.mkdir(dst_full_path)

            print("Computing {} for {} side".format(subj, side))
            extract_sulcal_pits(meshs[subj][side], dst_full_path, side=side, mask_path=masks[subj][side])

            cpt_subj += 1
        if cpt_subj == nb_subj:
            break
        print("Fin {}".format(subj))


if __name__ == "__main__":
    base_path = "/hpc/meca/data/REPRO_database"
    dst_path = "/envau/work/meca/users/leroux.b/output_REPRO_database_bis"

    sulcals_pits = os.path.join(base_path, "SulcalPits")
    meshs_path = os.path.join(base_path, "FS_database_OASIS_test_retest_FS5.3.0")

    subjects = get_subj(sulcals_pits)

    meshs = get_mesh(subjects, meshs_path)
    masks = get_mask(subjects, sulcals_pits)

    compute_all(meshs, masks, dst_path)


