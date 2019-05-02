import numpy as np
from slam import texture


def test_update_darray():
    test_texture = texture.TextureND()
    darray = np.zeros((4, 1))
    test_texture.update_darray(darray)
    assert test_texture.shape == darray.shape
