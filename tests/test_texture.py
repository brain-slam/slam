import numpy as np
from slam import texture
import unittest

class TestTextureMethods(unittest.TestCase):

    def test_update_darray(self):
        test_texture = texture.TextureND()
        darray = np.zeros((4, 1))
        test_texture.update_darray(darray)
        assert test_texture.shape == darray.shape

    def test_copy_texture(self):
        test_textureA = texture.TextureND()
        darray = np.zeros((4, 1))
        test_textureA.update_darray(darray)
        test_textureB = test_textureA.copy()
        assert (test_textureA.darray == test_textureB.darray).all()

    def test_copy_modif_texture(self):
        test_textureA = texture.TextureND()
        darrayA = np.zeros((4, 1))
        test_textureA.update_darray(darrayA)
        test_textureB = test_textureA.copy()
        test_textureB.darray[0][0] = 1
        assert not (test_textureA.darray == test_textureB.darray).all()


    def test_z_score_texture(self):
        darrayA = np.array([
            [5, 3, 2],
            [7, 2, 1],
            [4, 7, 4],
        ])
        outA = np.array([
            [3, 3, 3],
            [2, 2, 1],
            [4, 4, 4],
        ])

        test_textureA = texture.TextureND()
        test_textureA.update_darray(darrayA)
        test_textureA.z_score_filtering(1)
        assert test_textureA.darray.all() == outA.all()

if __name__ == '__main__':
    unittest.main()