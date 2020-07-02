import numpy as np
import unittest
import trimesh
import slam.curvature as scurv

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

            with self.subTest(i):

                # Sphere of radius i
                mesh_a = trimesh.creation.icosphere(
                    subdivisions=1, radius=float(i))

                curv, d1, d2 = scurv.curvatures_and_derivatives(mesh_a)

                shape = curv.shape

                # The curvature in every point is 1/radius
                final = np.full(shape, 1/i)

                assert(np.isclose(curv, final, precision_A).all())

        # Iterations on subdivisions

        for i in [1, 2, 3]:

            precision_A = .000001

            with self.subTest(i):

                # Sphere of radius 2, i subdivisions
                mesh_a = trimesh.creation.icosphere(subdivisions=i, radius=2)

                curv, d1, d2 = scurv.curvatures_and_derivatives(mesh_a)

                shape = curv.shape

                # The curvature in every point is 1/radius
                final = np.full(shape, 1/2)

                assert(np.isclose(curv, final, precision_A).all())


if __name__ == '__main__':
    unittest.main()
