import numpy as np
import nibabel as nb
import trimesh

from slam import texture


def load_mesh(gifti_file):
    """
    load gifti_file and create a trimesh object
    :param gifti_file: str, path to the gifti file on the disk
    :return: the corresponding trimesh object
    """
    g = nb.load(gifti_file)
    coords, faces = g.get_arrays_from_intent(
        nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data, \
        g.get_arrays_from_intent(
            nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
    metadata = g.meta.metadata
    metadata['filename'] = gifti_file
    return trimesh.Trimesh(faces=faces, vertices=coords,
                           metadata=metadata, process=False)


def write_mesh(mesh, gifti_file):
    """ Create a mesh object from two arrays

    fixme:  intent should be set !
    """
    coord = mesh.vertices
    triangles = mesh.faces
    carray = nb.gifti.GiftiDataArray(coord.astype(np.float32),
                                     "NIFTI_INTENT_POINTSET")
    tarray = nb.gifti.GiftiDataArray(
        triangles.astype(np.float32), "NIFTI_INTENT_TRIANGLE")
    img = nb.gifti.GiftiImage(darrays=[carray, tarray])
    # , meta=mesh.metadata)

    nb.save(img, gifti_file)


def load_texture(gifti_file):
    """
    load gifti_file and create a TextureND object (multidimensional)
    :param gifti_file: str, path to the gifti file on the disk
    :return: the corresponding TextureND object
    """
    # read the gifti usinng nibabel
    nb_texture = nb.gifti.read(gifti_file)
    # concatenate all the data arrays in a single numpy array
    cat_darrays = [nb_texture.darrays[i].data
                   for i in range(len(nb_texture.darrays))]
    return texture.TextureND(darray=np.array(cat_darrays),
                             metadata=nb_texture.meta.metadata)


def write_texture(tex, gifti_file):
    """
    TODO manage metadata
    write a TextureND object to disk as a gifti file
    :param gifti_file: str, path to the gifti file on the disk
    :return: the corresponding TextureND object
    """
    darrays_list = []
    for d in tex.darray:
        gdarray = nb.gifti.GiftiDataArray(
            d.astype(np.float32), 0)
        # gdarray.metadata = tex.metadata
        # print(gdarray.metadata)
        darrays_list.append(gdarray)
    out_texture_gii = nb.gifti.GiftiImage(darrays=darrays_list)
    # out_metadata = tex.metadata
    # print(out_metadata)
    # out_metadata['Name']=gifti_file
    # print(out_metadata)
    # out_texture_gii.set_metadata(nb.gifti.GiftiMetaData(out_metadata))
    # , meta=str(tex.metadata))

    nb.save(out_texture_gii, gifti_file)
