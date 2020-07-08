import numpy as np
import unittest
import trimesh
import slam.curvature as scurv
import slam.generate_parametric_surfaces as sgps

# UTILITIES


def make_sphere():
    """ Create a sphere"""
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return mesh_a


def make_box():
    """ Create a box"""
    mesh_a = trimesh.creation.box((1, 1, 1))
    return mesh_a


class TestCurvatureMethods(unittest.TestCase):

    # Sphere
    sphere_A = make_sphere()

    # Box
    box_A = make_box()

    def test_basic(self):
        mesh_a = self.sphere_A.copy()
        mesh_a_save = self.sphere_A.copy()

        cur, d1, d2 = scurv.curvatures_and_derivatives(mesh_a)

        # Non modification
        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        assert(mesh_a.faces == mesh_a_save.faces).all()

    def test_correctness_curvature(self):

        # Iterations on radius

        for i in [1, 2, 4]:

            precision_A = .000001
            precision_B = .000001

            with self.subTest(i):

                # Sphere of radius i
                mesh_a = trimesh.creation.icosphere(
                    subdivisions=1, radius=float(i))

                curv, d1, d2 = scurv.curvatures_and_derivatives(mesh_a)

                mean_curv = 0.5 * (curv[0, :] + curv[1, :])

                gauss_curv = (curv[0, :] * curv[1, :])

                shape = mean_curv.shape

                # The mean curvature in every point is 1/radius
                analytical_mean = np.full(shape, 1 / i)

                # The gaussian curvature in every point is 1/radius**2
                analytical_gauss = np.full(shape, 1 / i**2)

                assert(
                    np.isclose(
                        mean_curv,
                        analytical_mean,
                        precision_A).all())
                assert(
                    np.isclose(
                        gauss_curv,
                        analytical_gauss,
                        precision_B).all())

        # Iterations on subdivisions

        for i in [1, 2, 3]:

            precision_A = .000001
            precision_B = .000001

            with self.subTest(i):

                # Sphere of radius 2, i subdivisions
                mesh_a = trimesh.creation.icosphere(subdivisions=i, radius=2)

                curv, d1, d2 = scurv.curvatures_and_derivatives(mesh_a)

                mean_curv = 0.5 * (curv[0, :] + curv[1, :])
                gauss_curv = (curv[0, :] * curv[1, :])

                shape = mean_curv.shape

                # The curvature in every point is 1/radius
                analytical_mean = np.full(shape, 1 / 2)

                # The curvature in every point is 1/radius**2
                analytical_gauss = np.full(shape, 1 / 4)

                assert(
                    np.isclose(
                        mean_curv,
                        analytical_mean,
                        precision_A).all())
                assert(np.isclose(analytical_gauss,
                                  analytical_gauss, precision_B).all())

    @unittest.skip
    def test_correctness_curvature_low_error(self):

        K = [1, 1]
        quadric = sgps.generate_quadric(K, nstep=[20, 20], ax=3, ay=1,
                                        random_sampling=True,
                                        ratio=0.3,
                                        random_distribution_type='gamma')

        # Computation of estimated curvatures

        p_curv, d1, d2 = scurv.curvatures_and_derivatives(quadric)

        k1_estim, k2_estim = p_curv[0, :], p_curv[1, :]

        k_gauss_estim = k1_estim * k2_estim

        k_mean_estim = .5 * (k1_estim + k2_estim)

        # Computation of analytical curvatures

        k_mean_analytic = sgps.quadric_curv_mean(K)(
            np.array(quadric.vertices[:, 0]), np.array(quadric.vertices[:, 1]))

        k_gauss_analytic = sgps.quadric_curv_gauss(K)(
            np.array(quadric.vertices[:, 0]), np.array(quadric.vertices[:, 1]))

        k1_analytic = np.zeros((len(k_mean_analytic)))
        k2_analytic = np.zeros((len(k_mean_analytic)))

        for i in range(len(k_mean_analytic)):
            a, b = np.roots((1, -2 * k_mean_analytic[i], k_gauss_analytic[i]))
            k1_analytic[i] = min(a, b)
            k2_analytic[i] = max(a, b)

        # /// STATS

        k_mean_relative_change = abs(
            (k_mean_analytic - k_mean_estim) / k_mean_analytic)
        k_mean_absolute_change = abs((k_mean_analytic - k_mean_estim))

        k1_relative_change = abs((k1_analytic - k1_estim) / k1_analytic)
        k1_absolute_change = abs((k1_analytic - k1_estim))

        a = []
        a += [
            ["K_MEAN", "mean", "relative change", np.mean(
                k_mean_relative_change * 100), "%"],
            ("K_MEAN", "std", "relative change",
             np.std(k_mean_relative_change * 100), "%"),
            ("K_MEAN", "max", "relative change",
             np.max(k_mean_relative_change * 100), "%"),
            ["K_MEAN", "mean", "absolute change",
                np.mean(k_mean_absolute_change)],
            ["K_MEAN", "std", "absolute change",
                np.std(k_mean_absolute_change)],
            ["K_MEAN", "max", "absolute change",
                np.max(k_mean_absolute_change)],
            ("  K1", "mean", "relative change",
             np.mean(k1_relative_change * 100), "%"),
            ("  K1", "std", "relative change",
             np.std(k1_relative_change * 100), "%"),
            ("  K1", "max", "relative change",
             np.max(k1_relative_change * 100), "%"),
            ("  K1", "mean", "absolute change", np.mean(k_mean_absolute_change)),
            ("  K1", "std", "absolute change", np.std(k_mean_absolute_change)),
            ("  K1", "max", "absolute change", np.max(k_mean_absolute_change)),
        ]

        # PRINT STATS
        print("----------------------------------------")
        for i, v in enumerate(a):
            numeric_value = np.round(v[3], decimals=3)
            if i == 6:
                print("----------------------------------------")
            if len(v) > 4:
                print('{0:10} {1:5} {2:16} {3:2} {4} {5}'.format(
                    v[0], v[1], v[2], "=", numeric_value, v[4]))
            else:
                print('{0:10} {1:5} {2:16} {3:2} {4}'.format(
                    v[0], v[1], v[2], "=", numeric_value))
        print("----------------------------------------")

    @unittest.skip
    def test_correctness_curvature_drop_error(self):

        out = []

        for j in range(3):
            K = [1, 1]
            # Increase the number of points
            quadric = sgps.generate_quadric(
                K,
                nstep=[
                    20 + 10 * j,
                    20 + 10 * j],
                ax=3,
                ay=1,
                random_sampling=True,
                ratio=0.3,
                random_distribution_type='gamma')

            # Computation of estimated curvatures

            p_curv, d1, d2 = scurv.curvatures_and_derivatives(quadric)

            k1_estim, k2_estim = p_curv[0, :], p_curv[1, :]

            k_gauss_estim = k1_estim * k2_estim

            k_mean_estim = .5 * (k1_estim + k2_estim)

            # Computation of analytical curvatures

            k_mean_analytic = sgps.quadric_curv_mean(K)(
                np.array(quadric.vertices[:, 0]), np.array(quadric.vertices[:, 1]))

            k_gauss_analytic = sgps.quadric_curv_gauss(K)(
                np.array(quadric.vertices[:, 0]), np.array(quadric.vertices[:, 1]))

            k1_analytic = np.zeros((len(k_mean_analytic)))
            k2_analytic = np.zeros((len(k_mean_analytic)))

            for i in range(len(k_mean_analytic)):
                a, b = np.roots(
                    (1, -2 * k_mean_analytic[i], k_gauss_analytic[i]))
                k1_analytic[i] = min(a, b)
                k2_analytic[i] = max(a, b)

            # /// STATS

            k_mean_relative_change = abs(
                (k_mean_analytic - k_mean_estim) / k_mean_analytic)
            k_mean_absolute_change = abs((k_mean_analytic - k_mean_estim))

            k1_relative_change = abs((k1_analytic - k1_estim) / k1_analytic)
            k1_absolute_change = abs((k1_analytic - k1_estim))

            out += [np.mean(k_mean_absolute_change)]

        print(out)

        # Assert the absolute mean error decreases as we increase the number of
        # points
        assert(sorted(out, reverse=True) == out)


if __name__ == '__main__':

    unittest.main()
