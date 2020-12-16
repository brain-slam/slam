import unittest
import trimesh
import slam.differential_geometry as sdg
import numpy as np

tol = 1e-15


class TestDifferentialGeometry(unittest.TestCase):
    def test_gradient(self):
        # Trivial example at the moment:
        # Uniform texture: gradient vanishes
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        n_vert = mesh.vertices.shape[0]
        uniform_texture = np.ones((n_vert,))
        gradient_uniform = sdg.gradient(mesh, uniform_texture)
        self.assertTrue(
            (np.abs(gradient_uniform - np.zeros((n_vert, 3))) < tol).all())


if __name__ == '__main__':
    unittest.main()
