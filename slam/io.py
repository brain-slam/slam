import nibabel as nb
import trimesh
import numpy as np


def load(gifti_file):
    """
    load gifti_file and create a trimesh object
    :param gifti_file: str, path to the gifti file on the disk
    :return: the corresponding trimesh object
    """
    coords, faces = nb.gifti.read(gifti_file).getArraysFromIntent(
        nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data, \
        nb.gifti.read(gifti_file).getArraysFromIntent(
            nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
    return trimesh.Trimesh(faces=faces, vertices=coords, process=False)


def write(mesh, gifti_file):
    """ Create a mesh object from two arrays

    fixme:  intent should be set !
    """
    coord = mesh.faces
    triangles = mesh.vertices
    carray = nb.gifti.GiftiDataArray().from_array(coord.astype(np.float32),
                                                  "NIFTI_INTENT_POINTSET")
    tarray = nb.gifti.GiftiDataArray().from_array(
        triangles, "NIFTI_INTENT_TRIANGLE")
    img = nb.gifti.GiftiImage(darrays=[carray, tarray])

    nb.gifti.write(img, gifti_file)
