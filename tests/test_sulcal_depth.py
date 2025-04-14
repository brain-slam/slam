import numpy as np
import unittest
import slam.texture as stex
import slam.io as sio
import slam.sulcal_depth as sdepth

acceptable_error = 0.0001

class TestSulcalDepth(unittest.TestCase):
    test_mesh = sio.load_mesh('examples/data/example_mesh.gii')
    test_dpf = sio.load_texture('examples/data/example_dpf.gii').darray[0]
    test_dpf_star = sio.load_texture('examples/data/example_dpf_star.gii').darray[0]
    dpf = sdepth.depth_potential_function(test_mesh)[0]
    dpf_star = sdepth.dpf_star(test_mesh)[0]
    def test_basic(self):
        # test size
        self.assertTrue(self.test_dpf.shape==self.dpf.shape)
        self.assertTrue(self.test_dpf_star.shape == self.dpf_star.shape)

    def test_consistency(self):
        self.assertTrue(np.max(np.abs(self.test_dpf-self.dpf)) < acceptable_error)
        self.assertTrue(np.max(np.abs(self.test_dpf_star - self.dpf_star)) < acceptable_error)

if __name__ == "__main__":

    unittest.main()
