"""
Unit test for the spangy module
TODO: everything
"""

import unittest
import trimesh


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
        """
        mesh = self.sphere_test.copy()
        # Save mesh to disk in .mesh format
        sio.write_mesh(mesh, '../spangy/Data/test_spangy_eigenpairs.gii')
        # Compute the eigenpairs, for N = 20
        N = 20
        eigVal, eigVects, lap_b = spgy.spangy_eigenpairs(mesh, N)

        # Assert the results are correct
        eigVal_corr = [1.33235160658447e-16,
                       2.18647331318054,
                       2.18647331318055,
                       2.18647331318055,
                       7.16750557270010,
                       7.16750557270011,
                       7.16750558409602,
                       7.16750558409602,
                       16.1774872478754,
                       16.1774872478754]
        print(eigVal)
        assert np.isclose(eigVal, eigVal_corr, self.precision).all()
        """
        assert 2 == 2

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
