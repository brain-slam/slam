import slam.distortion as sdst
import numpy as np
import unittest
import trimesh

# UTILITIES


def make_sphere(radius=1):
    """ Create a sphere"""
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    return mesh_a


class TestDistortionMethods(unittest.TestCase):

    # Spheres
    sphere_r1 = make_sphere(1)
    sphere_r2 = make_sphere(2)

    def test_basic(self):
        mesh_a = self.sphere_r1.copy()
        mesh_a_save = self.sphere_r1.copy()

        mesh_b = self.sphere_r2.copy()
        mesh_b_save = self.sphere_r2.copy()

        # tmp = sdst.angle_difference(mesh_a, mesh_b)

        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        assert(mesh_a.faces == mesh_a_save.faces).all()

        assert(mesh_b.vertices == mesh_b_save.vertices).all()
        assert(mesh_b.faces == mesh_b_save.faces).all()

        # tmp = sdst.area_difference(mesh_a, mesh_b)

        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        assert(mesh_a.faces == mesh_a_save.faces).all()

        assert(mesh_b.vertices == mesh_b_save.vertices).all()
        assert(mesh_b.faces == mesh_b_save.faces).all()

        # tmp = sdst.edge_length_difference(mesh_a, mesh_b)

        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        assert(mesh_a.faces == mesh_a_save.faces).all()

        assert(mesh_b.vertices == mesh_b_save.vertices).all()
        assert(mesh_b.faces == mesh_b_save.faces).all()

    def test_correctness_area(self):

        precisionA = .001

        n_sub = 5

        mesh_a = trimesh.creation.icosphere(subdivisions=n_sub, radius=1)
        mesh_b = trimesh.creation.icosphere(subdivisions=n_sub, radius=2)

        area_diff_estim = sdst.area_difference(mesh_a, mesh_b)

        area_diff_estim = sum(abs(area_diff_estim))

        area_diff_analytical = 4 * np.pi * 2 * 2 - 4 * np.pi * 1 * 1

        assert(np.isclose(area_diff_estim, area_diff_analytical, precisionA))

    def test_correctness_angle(self):

        precisionA = .001

        n_sub = 5

        mesh_a = trimesh.creation.icosphere(subdivisions=n_sub, radius=1)
        mesh_b = trimesh.creation.icosphere(subdivisions=n_sub, radius=2)

        angle_diff_estim = sdst.angle_difference(mesh_a, mesh_b)

        angle_diff_estim = sum(sum(abs(angle_diff_estim)))

        angle_diff_analytical = 0

        assert(np.isclose(angle_diff_estim, angle_diff_analytical, precisionA))


if __name__ == '__main__':
    unittest.main()
