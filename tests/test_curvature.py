import numpy as np
import unittest
import trimesh
import slam.curvature as scurv
import slam.generate_parametric_surfaces as sgps
import slam.utils as ut

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

    def test_local_basis(self):
        precision = 1e-12
        # First test, generic situation
        normal1 = 1/np.sqrt(3)* np.array([[1], [1], [1]])
        e1 = 1/np.sqrt(6) * np.array([[2], [-1], [-1]])
        e2 = 1/np.sqrt(2) * np.array([[0], [1], [-1]])
        res = scurv.determine_local_basis(normal1, precision)
        assert(np.isclose(res[0], e1, precision).all())
        assert(np.isclose(res[1], e2, precision).all())

        # Second test, when norm(vec1) < tol in determine_local_basis
        normal1 = np.array([[1], [0], [0]])
        e1 = np.array([[0],[1],[0]])
        e2 = np.array([[0],[0],[1]])
        res = scurv.determine_local_basis(normal1, precision)
        assert(np.isclose(res[0], e1, precision).all())
        assert(np.isclose(res[1], e2, precision).all())


    def test_correctness_curvature_sphere(self):

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
                assert(
                    np.isclose(
                        analytical_gauss,
                        analytical_gauss,
                        precision_B).all())

    # @unittest.skip
    def test_correctness_curvature_quadric(self):

        K = [1, 1]
        quadric = sgps.generate_quadric(K, nstep=[20, 20], ax=3, ay=1,
                                        random_sampling=True,
                                        ratio=0.3,
                                        random_distribution_type='gamma')

        # Computation of estimated curvatures

        p_curv, d1, d2 = scurv.curvatures_and_derivatives(quadric)

        k1_estim, k2_estim = p_curv[0, :], p_curv[1, :]

        # k_gauss_estim = k1_estim * k2_estim

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
        # k1_absolute_change = abs((k1_analytic - k1_estim))

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
            (" K1", "mean", "relative change",
             np.mean(k1_relative_change * 100), "%"),
            (" K1", "std", "relative change",
             np.std(k1_relative_change * 100), "%"),
            (" K1", "max", "relative change",
             np.max(k1_relative_change * 100), "%"),
            (" K1", "mean", "absolute change",
             np.mean(k_mean_absolute_change)),
            (" K1", "std", "absolute change", np.std(k_mean_absolute_change)),
            (" K1", "max", "absolute change", np.max(k_mean_absolute_change)),
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

    def test_error_drop_curvature_quadric(self):
        """ Asserts the error drops when the mesh resolution is increased"""

        out = []

        for j in range(3):
            K = [1, 1]

            # Increase of the number of points
            quadric = sgps.generate_quadric(
                K,
                nstep=[
                    20 + 10 * j,
                    20 + 10 * j],
                ax=3,
                ay=1,
                random_sampling=False,
                ratio=0.3,
                random_distribution_type='gamma')

            # Computation of estimated curvatures

            p_curv, d1, d2 = scurv.curvatures_and_derivatives(quadric)

            k1_estim, k2_estim = p_curv[0, :], p_curv[1, :]

            # k_gauss_estim = k1_estim * k2_estim

            k_mean_estim = .5 * (k1_estim + k2_estim)

            # Computation of analytical curvatures

            k_mean_analytic = sgps.quadric_curv_mean(K)(
                np.array(quadric.vertices[:, 0]),
                np.array(quadric.vertices[:, 1]))

            k_gauss_analytic = sgps.quadric_curv_gauss(K)(
                np.array(quadric.vertices[:, 0]),
                np.array(quadric.vertices[:, 1]))

            k1_analytic = np.zeros((len(k_mean_analytic)))
            k2_analytic = np.zeros((len(k_mean_analytic)))

            for i in range(len(k_mean_analytic)):
                a, b = np.roots(
                    (1, -2 * k_mean_analytic[i], k_gauss_analytic[i]))
                k1_analytic[i] = min(a, b)
                k2_analytic[i] = max(a, b)

            # /// STATS

            # k_mean_relative_change = abs(
            #     (k_mean_analytic - k_mean_estim) / k_mean_analytic)
            k_mean_absolute_change = abs((k_mean_analytic - k_mean_estim))

            # k1_relative_change = abs((k1_analytic - k1_estim) / k1_analytic)
            # k1_absolute_change = abs((k1_analytic - k1_estim))

            out += [np.round(np.mean(k_mean_absolute_change), 6)]

        # Assert the absolute mean error decreases as we increase the number of
        # points
        assert(sorted(out, reverse=True) == out)

    def test_error_drop_curvature_sphere(self):

        out = []

        for j in range(3):
            mesh_a = trimesh.creation.icosphere(
                subdivisions=j + 1, radius=2)
            p_curv, d1, d2 = scurv.curvatures_and_derivatives(mesh_a)
            k1_estim, k2_estim = p_curv
            k_mean_estim = (k1_estim + k2_estim) * .5
            # k_gauss_estim = (k1_estim * k2_estim)

            out += [np.round(np.mean(k_mean_estim), 6)]

        assert(sorted(out, reverse=True) == out)

    def test_correctness_decomposition_sphere(self):

        precisionA = .0000001
        precisionB = .0000001

        radius = 3

        mesh_a = trimesh.creation.icosphere(
            subdivisions=2, radius=radius)

        mesh_a_save = mesh_a.copy()

        shapeIndex, curvedness = scurv.curvedness_shapeIndex(mesh_a)

        # Non modification

        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        assert(mesh_a.faces == mesh_a_save.faces).all()

        # Correctness
        # ShapeIndex is 1 or -1 on every vertex
        # Curvedness is 1/radius on every vertex
        assert(
            np.isclose(
                shapeIndex, 1, precisionA) | np.isclose(
                shapeIndex, -1, precisionA)).all()
        assert(np.isclose(curvedness, 1 / radius, precisionB).all())

    def test_correctness_decomposition_quadric(self):

        # Precision on the shapeIndex calculation
        precisionA = 0.000001
        # Precision on the curvedness calculation
        precisionB = 0.000001

        set_of_tests = [
            # Set = [K, correct shapeIndex, correct curvedness]
            # For example, on a quadric generated by K=[1,1], the shapeIndex
            # is +-1 at the center and the curvedness is +-2
            [
                [1, 1], 1, 2
            ],

            [
                [-1, -1], 1, 2
            ],
            [
                [.5, .5], 1, 1
            ],
            [
                [1, 0], .5, np.sqrt(2)
            ],
        ]

        for i in range(len(set_of_tests)):

            current_test = set_of_tests[i]

            K = current_test[0]

            # Correct values of shapeIndex and curvedness on the vertex at the
            # center of the mesh
            correct_shape_index = current_test[1]
            correct_curvedness = current_test[2]

            with self.subTest(i):

                # Generate quadric
                quadric = sgps.generate_quadric(K, nstep=[20, 20], ax=3, ay=1,
                                                random_sampling=False,
                                                ratio=0.3,)

                # Computation of analytical k_mean, k_gauss, k_1 and k_2
                k_mean_analytic = sgps.quadric_curv_mean(K)(
                    np.array(quadric.vertices[:, 0]),
                    np.array(quadric.vertices[:, 1]))

                k_gauss_analytic = sgps.quadric_curv_gauss(K)(
                    np.array(quadric.vertices[:, 0]),
                    np.array(quadric.vertices[:, 1]))

                k1_analytic = np.zeros((len(k_mean_analytic)))
                k2_analytic = np.zeros((len(k_mean_analytic)))

                for i in range(len(k_mean_analytic)):
                    a, b = np.roots(
                        (1, -2 * k_mean_analytic[i], k_gauss_analytic[i]))
                    k1_analytic[i] = min(a, b)
                    k2_analytic[i] = max(a, b)

                # Decomposition of the curvature
                shapeIndex, curvedness = scurv.decompose_curvature(
                    np.array((k1_analytic, k2_analytic)))

                # Find the index of the vertex which is the closest to the
                # center

                def mag(p):
                    """Distance to the center for the point p, not considering
                     the z axis"""
                    x = p[0]
                    y = p[1]
                    return np.sqrt(x * x + y * y)

                min_i = 0
                min_m = mag(quadric.vertices[0])
                for i, v in enumerate(quadric.vertices):
                    mg = mag(v)
                    if mg < min_m:
                        min_i = i
                        min_m = mg

                # Correctness
                computed_shapeIndex = shapeIndex[min_i]
                computed_curvedness = curvedness[min_i]

                assert(
                    np.isclose(
                        computed_shapeIndex, correct_shape_index,
                        precisionA) | np.isclose(
                        computed_shapeIndex, -correct_shape_index, precisionA))
                assert(
                    np.isclose(
                        computed_curvedness, correct_curvedness,
                        precisionB) | np.isclose(
                        computed_curvedness, -correct_curvedness, precisionB))

    def test_correctness_direction_quadric(self):

        # WARNING: THRESHOLD should be st to 15°, here this value is just to
        # ensure the test is okay
        # NEED to check Rusinkiewicz method and the proper orientation of
        # curvature directions
        THRESHOLD = 10

        # Generate a paraboloid
        K = [1, 0]

        # Increase of the number of points
        quadric = sgps.generate_quadric(
            K,
            nstep=[
                20,
                20],
            ax=3,
            ay=3,
            random_sampling=False,
            ratio=0.3,
            random_distribution_type='gamma', equilateral=True)

        # ESTIMATED Principal curvature, Direction1, Direction2
        p_curv_estim, d1_estim, d2_estim = scurv.curvatures_and_derivatives(
            quadric)

        # ANALYTICAL directions
        analytical_directions = sgps.compute_all_principal_directions_3D(
            K, quadric.vertices)

        estimated_directions = np.zeros(analytical_directions.shape)
        estimated_directions[:, :, 0] = d1_estim
        estimated_directions[:, :, 1] = d2_estim

        angular_error_0, dotprods = ut.compare_analytic_estimated_directions(
            analytical_directions[:, :, 0], d2_estim, abs=True)
        angular_error_0 = 180 * angular_error_0 / np.pi

        # CORRECTNESS DIRECTION 1

        # Number of vertices where the angular error is lower than 20 degrees
        n_low_error = np.sum(angular_error_0 < THRESHOLD)

        # Percentage of vertices where the angular error is lower than 20
        # degrees
        percentage_low_error = (n_low_error / angular_error_0.shape[0])

        assert(percentage_low_error > .75)

        # CORRECTNESS DIRECTION 2

        angular_error_1, dotprods = ut.compare_analytic_estimated_directions(
            analytical_directions[:, :, 1], d1_estim, abs=True)
        angular_error_1 = 180 * angular_error_1 / np.pi

        # Number of vertices where the angular error is lower than 20 degrees
        n_low_error = np.sum(angular_error_1 < THRESHOLD)

        # Percentage of vertices where the angular error is lower than 20
        # degrees
        percentage_low_error = (n_low_error / angular_error_1.shape[0])

        assert(percentage_low_error > .75)


if __name__ == '__main__':

    unittest.main()
