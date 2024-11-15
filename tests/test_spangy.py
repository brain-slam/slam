"""
Unit test for the spangy module
TODO: everything
"""
import unittest
import trimesh
import numpy as np
import slam.spangy as spgy
import slam.differential_geometry as sdg
# import slam.curvature as scurv


def make_sphere():
    """ Create a sphere"""
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return mesh_a


def make_rectangle():
    """ Create a rectangle"""
    mesh_b = trimesh.creation.box(extents=[1.0, 2.0, 0.5])
    return mesh_b


class TestSpangy(unittest.TestCase):

    # Sphere
    sphere_test = make_sphere()
    rectangle_test = make_rectangle()
    precision = 1e-6

    def test_spangy_eigenpairs(self):
        """
        Test that the eigenvalues generation is correct, for a simple case.
        We compare the obtained results to the ones obtained with the matlab
        code. Which we assume is correct.
        """
        mesh = self.rectangle_test.copy()

        # Compute the eigenpairs, for N = 5
        N = 5
        eigVal, eigVects, _ = spgy.eigenpairs(mesh, N)
        # Compute the mesh laplacian outside
        lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='fem')
        assert np.allclose(lap @ eigVects - eigVal * (lap_b @ eigVects),
                           np.zeros((mesh.vertices.shape[0], N)
                                    ), atol=self.precision)
        # Compared to matlab results (hardcoded here for the same rectangle)
        eigVal_matlab = [1.31874474854710e-15,
                         1.99766647468559,
                         5.37024944254656,
                         7.38658922693569,
                         17.0853208854239]
        assert np.allclose(eigVal, eigVal_matlab, atol=self.precision)

    def test_spangy_spectrum(self):
        """
        Test that the spectrum generation is correct, for a simple case.

        We compare the obtained results to the ones obtained with the
        matlab code.
        which we assume is correct.
        """
        """
        mesh = self.rectangle_test.copy()
        # Compute the eigenpairs, for N = 5
        N = 5
        eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

        # Compute the curvature
        PrincipalCurvatures, _, _ = scurv.curvatures_and_derivatives(mesh)
        mean_curv = 0.5 * \
            (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

        grouped_spectrum, group_indices, _ = spgy.spectrum(mean_curv, lap_b,
                                                           eigVects, eigVal)
        levels = len(group_indices)
        assert levels == 3

        # Compared to matlab results (hardcoded here for the same rectangle)
        grouped_spectrum_matlab = [5.333333, 1.85197e-30, 4.930e-32]
        assert np.allclose(grouped_spectrum, grouped_spectrum_matlab,
                           atol=self.precision)
        """

    def test_spangy_local_dominance_map(self):
        """
        Test the output of the local dominance maps for a simple case.

        We compare the obtained results to the ones obtained with the
        matlab code.
        which we assume is correct.
        """

        """
        Test with a rectangle not the best idea for this test
        suggestions?

        mesh = self.rectangle_test.copy()
        # Compute the eigenpairs, for N = 5
        N = 5
        eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

        # Compute the curvature
        PrincipalCurvatures, _, _ = scurv.curvatures_and_derivatives(mesh)
        mean_curv = 0.5 * (PrincipalCurvatures[0, :] +
                           PrincipalCurvatures[1, :])

        grouped_spectrum, group_indices, coefficients = spgy.spectrum(
                                                             mean_curv, lap_b,
                                                             eigVects, eigVal)
        levels = len(group_indices)
        # eigVects = np.flip(eigVects, 1)
        # Compute the local dominance map
        loc_dom_band, frecomposed = spgy.local_dominance_map(coefficients,
                                                             mean_curv,
                                                             levels,
                                                             group_indices,
                                                             eigVects)

        # Compared to matlab results (hardcoded here for the same rectangle)
        # results differ! (sign issue)
        loc_dom_band_matlab = [1, 1, 2, 2, 1, 1, 2, 2]
        frecomposed_matlab = np.array([[4.92517e-16, -2.23134e-16],
                                       [2.34773e-16, -5.4280402e-17],
                                       [-5.4702e-16, 5.42804e-17],
                                       [-8.6377e-16, 2.23134e-16],
                                       [1.18850e-15, 5.428040e-17],
                                       [9.0114e-16, 2.2313e-16],
                                       [-5.29881e-16, -2.2313e-16],
                                       [-8.7624e-16, -5.428e-17]])
        import pdb; pdb.set_trace()
        assert np.allclose(frecomposed, frecomposed_matlab,
                           atol=self.precision)
        assert np.allclose(loc_dom_band, loc_dom_band_matlab,
                           atol=self.precision)
        """


if __name__ == "__main__":
    unittest.main()
