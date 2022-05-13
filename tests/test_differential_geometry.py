import unittest

import numpy as np
import trimesh

import slam.differential_geometry as sdg
import slam.generate_parametric_surfaces as sgpm

TOL = 1e-15
TOL2 = 1e-1


def cartesian_to_spherical(coords):
    n = len(coords)
    spherical_coordinates = np.zeros((n, 3))
    for i in range(n):
        R = np.sqrt(np.sum(coords[i, :] ** 2))
        spherical_coordinates[i, 0] = R
        coord_i = coords[i, :] / R
        spherical_coordinates[i, 2] = np.arctan2(
            coord_i[1], coord_i[0]
        )  # Phi, in [0,2pi]
        spherical_coordinates[i, 1] = np.arccos(coord_i[2])  # Theta, in [0,pi]
    return spherical_coordinates


class TestDifferentialGeometry(unittest.TestCase):
    # def test_gradient(self):
    #     # Trivial example at the moment:
    #     # Uniform texture: gradient vanishes
    #     mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    #     n_vert = mesh.vertices.shape[0]
    #     uniform_texture = np.ones((n_vert,))
    #     gradient_uniform = sdg.gradient(mesh, uniform_texture)
    #     self.assertTrue(
    #         (np.abs(gradient_uniform - np.zeros((n_vert, 3))) < TOL).all())

    def test_gradient(self):
        sphere_mesh = sgpm.generate_sphere_icosahedron(subdivisions=4)
        spherical_coordinates = cartesian_to_spherical(sphere_mesh.vertices)
        phi = spherical_coordinates[:, 2]
        theta = spherical_coordinates[:, 1]
        gradient_theta = sdg.gradient_fast(sphere_mesh, theta)

        analytic_gradient_theta = np.zeros((sphere_mesh.vertices.shape[0], 3))
        analytic_gradient_theta[:, 0] = np.cos(theta) * np.cos(phi)
        analytic_gradient_theta[:, 1] = np.cos(theta) * np.sin(phi)
        analytic_gradient_theta[:, 2] = -np.sin(theta)
        error_norms = np.sqrt(
            np.sum((gradient_theta - analytic_gradient_theta) ** 2, axis=1)
        )
        # error_norms decrease linearly with mesh spacing
        self.assertTrue(np.mean(error_norms) < TOL2)


if __name__ == "__main__":
    unittest.main()
