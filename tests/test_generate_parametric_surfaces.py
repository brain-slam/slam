from slam import generate_parametric_surfaces as sps
import unittest

class TestGenerateParametricSurfacesMethods(unittest.TestCase):

    def test_generate_ellipsoid(self):
        nstep = 50
        randomSampling = True
        a = 2
        b = 1
        ellips = sps.generate_ellipsiod(a, b, nstep, randomSampling)
        assert len(ellips.vertices) == nstep**2

if __name__ == '__main__':
    unittest.main()