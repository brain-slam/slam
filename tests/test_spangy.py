"""
Unit test for the spangy module
TODO: everything
"""
import unittest
import trimesh
import numpy as np
import slam.spangy as spgy
import slam.differential_geometry as sdg


def make_sphere():
    """ Create a sphere"""
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return mesh_a


class TestSpangy(unittest.TestCase):

    # Sphere
    sphere_test = make_sphere()
    precision = 1e-6

    def test_spangy_eigenpairs(self):
        """
        Test that the eigenvalues generation is correct, for a simple case.

        We compare the obtained results to the ones obtained with the matlab
        code.
        which we assume is correct.
        """
        mesh = self.sphere_test.copy()

        # Compute the eigenpairs, for N = 20
        N = 20
        eigVal, eigVects, _ = spgy.eigenpairs(mesh, N)

        # Compute the mesh laplacian outside
        lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='fem')
        assert np.allclose(lap @ eigVects - eigVal * (lap_b @ eigVects),
                           np.zeros((mesh.vertices.shape[0], N)
                                    ), atol=self.precision)

        # TODO: some extra tests could be added

    def test_spangy_spectrum(self):
        """
        Test that the spectrum generation is correct, for a simple case.

        We compare the obtained results to the ones obtained with the
        matlab code.
        which we assume is correct.
        """
        print('nyi')
        assert 2 == 2

    def test_spangy_local_dominance_map(self):
        """
        Test the output of the local dominance maps for a simple case.

        We compare the obtained results to the ones obtained with the
        matlab code.
        which we assume is correct.
        """
        print('nyi')
        assert 2 == 2


if __name__ == "__main__":
    unittest.main()
