import numpy as np
import unittest
from slam import texture

class TestTexture(unittest.TestCase):
    def test_update_darray(self):
        test_texture = texture.TextureND()
        darray = np.zeros((4, 1))
        test_texture.update_darray(darray)
        assert test_texture.shape == darray.shape

if __name__ == "__main__":

    unittest.main()
