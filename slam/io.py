import nibabel as nb
import trimesh
import numpy as np
from slam import texture


def load_mesh(gifti_file):
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


def write_mesh(mesh, gifti_file):
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


def load_texture(gifti_file):
    """
    load gifti_file and create a TextureND object
    :param gifti_file: str, path to the gifti file on the disk
    :return: the corresponding TextureND object
    """
    nb_texture = nb.gifti.read(gifti_file)

    return texture.TextureND(darray=nb_texture.darrays[0].data)


def write_texture(tex, gifti_file):
    """
    write a TextureND object to disk as a gifti file
    :param gifti_file: str, path to the gifti file on the disk
    :return: the corresponding TextureND object
    """
    darrays_list = []
    outtexture_data = np.copy(tex.darray)  # or whatever you want to write

    darrays_list.append(nb.GiftiDataArray().from_array(
        outtexture_data.astype(np.float32), 0))
    outtexture_gii = nb.GiftiImage(darrays=darrays_list)

    nb.write(outtexture_gii, gifti_file)
