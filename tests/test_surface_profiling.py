import unittest
import numpy as np
import trimesh

from slam.surface_profiling import (
    surface_profiling_vert,
    compute_profiles_sampling_points,
    select_points_orientation,
    radial_sort,
    compute_profile_coord_x_y,
    compute_profile_barycentric_para,
    compute_profile_texture_barycentric,
    second_round_profiling_vert,
    vert2poly_indices,
    get_texture_value_on_profile,
    cortical_surface_profiling
)
from slam.texture import TextureND


def make_sphere(radius=1.0, subdivisions=1):
    """Create a sphere mesh for testing."""
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


class TestSurfaceProfiling(unittest.TestCase):

    def setUp(self):
        self.mesh = make_sphere(radius=1.0)
        self.vertex_idx = np.argmax(self.mesh.vertices[:, 1])
        self.vertex = self.mesh.vertices[self.vertex_idx]
        self.normal = self.mesh.vertex_normals[self.vertex_idx]

    def test_cortical_surface_profiling(self):
        """Test cortical surface profiling functionality.

        Verifies the computation of surface profiles on a unit sphere,
        checking the output shapes and expected coordinates.
        """
        rot_angle = 90.0  # Simpler angle for verification
        r_step = 0.1
        max_samples = 3

        profile_x, profile_y = cortical_surface_profiling(
            self.mesh, rot_angle, r_step, max_samples
        )

        # Number of profiles per vertex
        expected_profiles = int(360 / rot_angle)
        n_vertices = len(self.mesh.vertices)
        expected_shape = (n_vertices, expected_profiles, max_samples)
        self.assertEqual(profile_x.shape, expected_shape)
        self.assertEqual(profile_y.shape, expected_shape)

        # Test specific values for the top vertex (0,1,0)
        top_vertex_idx = np.argmax(self.mesh.vertices[:, 1])
        # For a unit sphere, x coordinates must be roughly r_step multiples
        expected_x_steps = np.array([0.1, 0.2, 0.3])  # r_step multiples
        np.testing.assert_array_almost_equal(
            profile_x[top_vertex_idx, 0, :], expected_x_steps, decimal=1
        )

        # SImilar idea to expected_x_steps, but for y, starting from 0
        # Bit hacky, but should be reproducible
        expected_y = np.array([-0.03, -0.06, -0.09])
        np.testing.assert_array_almost_equal(
            profile_y[top_vertex_idx, 0, :], expected_y, decimal=2
        )

    def test_surface_profiling_vert(self):
        """Test profiling for a single vertex.

        Verifies the computation of profiles around a single vertex,
        checking output shapes and dimensions.
        TODO: Check the actual profile output
        """
        init_rot_dir = np.array([1, 1, 1]) - self.vertex
        rot_angle = 90.0
        r_step = 1.0
        max_samples = 1

        profiles = surface_profiling_vert(
            self.vertex,
            self.normal,
            init_rot_dir,
            rot_angle,
            r_step,
            max_samples,
            self.mesh,
        )

        # Check output shape
        expected_profiles = int(360 / rot_angle)

        self.assertEqual(profiles.shape[0], expected_profiles)
        self.assertEqual(profiles.shape[1], max_samples)
        self.assertEqual(profiles.shape[2], 3)  # 3D coordinates

    def test_compute_profiles_sampling_points(self):
        """Test computation of profile sampling points.

        Verifies the generation of sampling points along profiles,
        checking output dimensions and point structure.
        Only check the shape, not the content, could
        be improved
        """
        points_intersect = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        origin = np.array([0, 0, 0])
        max_samples = 5
        r_step = 0.2

        profile_points = compute_profiles_sampling_points(
            points_intersect, origin, max_samples, r_step
        )

        self.assertEqual(len(profile_points), max_samples)
        self.assertTrue(all(len(point) == 3 for point in profile_points))

    def test_radial_sort(self):
        """Test radial sorting of points around an origin.

        Verifies that points are correctly sorted radially around an origin
        point, maintaining their distances while organizing them in angular
        order.
        """
        points = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        origin = np.array([0, 0, 0])
        normal = np.array([0, 0, 1])

        sorted_points, sorted_indices = radial_sort(points, origin, normal)

        self.assertEqual(len(sorted_points), len(points))
        self.assertEqual(len(sorted_indices), len(points))
        # Points should maintain their distance from origin
        original_distances = np.linalg.norm(points - origin, axis=1)
        sorted_distances = np.linalg.norm(sorted_points - origin, axis=1)
        np.testing.assert_array_almost_equal(
            sorted_distances,
            original_distances
        )

    def test_compute_profile_coord_x_y(self):
        """Test computation of profile coordinates in x-y plane.

        Verifies the projection and computation of profile coordinates
        onto a tangent plane defined by origin and normal vector.
        """
        profiles = np.array(
            [
                [[1, 0, 0], [2, 0, 0]],  # First profile
                [[0, 1, 0], [0, 2, 0]],  # Second profile
            ]
        )
        origin = np.array([0, 0, 0])
        normal = np.array([0, 0, 1])

        x, y = compute_profile_coord_x_y(profiles, origin, normal)

        # Check shapes
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(y.shape, (2, 2))

        # For these profiles, x should match the distance from origin in the
        # tangent plane
        np.testing.assert_array_almost_equal(x[0], [1, 2])  # First profile
        np.testing.assert_array_almost_equal(x[1], [1, 2])  # Second profile

        # y should be 0 as all points are in the tangent plane
        np.testing.assert_array_almost_equal(y, np.zeros((2, 2)))

    def test_second_round_profiling_vert(self):
        """Test second round of vertex profiling.

        Verifies the extended profiling computation that includes face indices
        and additional geometric information.
        """
        init_rot_dir = np.array([1, 1, 1]) - self.vertex
        rot_angle = 90.0
        r_step = 0.1
        max_samples = 3
        mesh_face_index = np.arange(len(self.mesh.faces))

        profile_points, face_indices = second_round_profiling_vert(
            self.vertex,
            self.normal,
            init_rot_dir,
            rot_angle,
            r_step,
            max_samples,
            self.mesh,
            mesh_face_index,
        )

        # Check output shapes
        expected_profiles = int(360 / rot_angle)
        self.assertEqual(profile_points.shape[0], expected_profiles)
        self.assertEqual(profile_points.shape[1], max_samples)
        self.assertEqual(profile_points.shape[2], 3)  # [p1, p2, sample_point]
        self.assertEqual(profile_points.shape[3], 3)  # 3D coordinates

        # Check face indices shape
        self.assertEqual(face_indices.shape[0], expected_profiles)
        self.assertEqual(face_indices.shape[1], max_samples)

    def test_vert2poly_indices(self):
        """Test vertex to polygon index mapping.

        Verifies the correct identification of polygons containing
        specified vertices in the mesh.
        """
        poly_array = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        vertex_array = np.array([3, 4])

        result = vert2poly_indices(vertex_array, poly_array)

        # Vertices 3 and 4 should be in polygons 1 and 2
        expected = np.array([1, 2])
        np.testing.assert_array_equal(np.sort(result), expected)

    def test_get_texture_value_on_profile(self):
        """Test retrieval of texture values along profiles.

        Verifies the correct sampling of texture values at profile points
        using mesh and texture information.

        Could probably design a better test, with texture being
        not all ones
        """
        texture = np.ones((len(self.mesh.vertices),))

        # Create sample profile points
        # 1 vertex, 2 profiles, 2 samples
        profile_samples = np.zeros((1, 2, 2, 3, 3))
        profile_samples_fid = np.zeros(
            (1, 2, 2), dtype=int
        )  # Corresponding face indices

        # Create proper TextureND object instead of mock
        texture_obj = TextureND(darray=texture)

        result = get_texture_value_on_profile(
            texture_obj, self.mesh, profile_samples, profile_samples_fid
        )

        # Should match input profile shape
        self.assertEqual(result.shape, (1, 2, 2))

        # should be all ones, as the original texture is all ones
        np.testing.assert_array_equal(result, np.ones((1, 2, 2)))

    def test_compute_profile_barycentric_para(self):
        """Test computation of barycentric coordinates for profile points.

        Verifies the calculation of barycentric coordinates for profile points
        within their respective triangles.
        """
        # 1 vertex, 2 profiles, 2 samples
        profile_points = np.zeros((1, 2, 2, 3, 3))
        triangle_id = np.zeros((1, 2, 2), dtype=int)

        barycentric = compute_profile_barycentric_para(
            profile_points, self.mesh, triangle_id
        )

        # Check output shape
        self.assertEqual(
            barycentric.shape, (1, 2, 2, 3)
        )  # Last dimension is barycentric coords

    def test_compute_profile_texture_barycentric(self):
        """Test computation of texture values using barycentric coordinates.

        Verifies the interpolation of texture values at profile points
        using barycentric coordinates within triangles.
        """
        texture = np.ones(len(self.mesh.vertices))
        # 1 vertex, 2 profiles, 2 samples
        triangle_id = np.zeros((1, 2, 2), dtype=int)
        # Equal barycentric coordinates
        barycentric_coord = np.ones((1, 2, 2, 3)) / 3

        result = compute_profile_texture_barycentric(
            texture, self.mesh, triangle_id, barycentric_coord
        )

        # Check output shape
        self.assertEqual(result.shape, (1, 2, 2))

        # For uniform texture and equal barycentric coordinates, result should
        # be 1
        np.testing.assert_array_almost_equal(result, np.ones((1, 2, 2)))

    def test_select_points_orientation(self):
        """Test selection and orientation of intersection points.

        Verifies the correct selection and ordering of intersection points
        with a simple example
        """
        intersect_points = np.array(
            [
                [[0, 0, 0], [1, 0, 0]],  # Line 1
                [[0, 0, 0], [0, 1, 0]],  # Line 2
                [[0, 0, 0], [-1, 0, 0]],  # Line 3 (opposite to Line 1)
                [[0, 0, 0], [0, -1, 0]],  # Line 4 (opposite to Line 2)
            ]
        )

        origin = np.array([0, 0, 0])
        normal = np.array([0, 0, 1])
        # Direction vector pointing towards positive x
        r_alpha = np.array([1, 0, 0])

        # TODO: check all outputs
        orient_points, _, lines_indices = select_points_orientation(
            intersect_points, r_alpha, origin, normal
        )

        # Check that we get points in positive x direction
        self.assertTrue(
            np.all(orient_points[:, 0] >= 0)
        )  # All x coordinates should be positive

        # Check shapes
        self.assertEqual(orient_points.shape[1], 3)  # Each point should be 3D

        # Check that points are ordered by distance from origin
        distances = np.linalg.norm(orient_points - origin, axis=1)
        self.assertTrue(
            np.all(np.diff(distances) >= 0)
        )  # Distances should be non-decreasing

        # Check that indices are valid
        self.assertTrue(np.all(lines_indices >= 0))
        self.assertTrue(np.all(lines_indices < len(intersect_points)))


if __name__ == "__main__":
    unittest.main()
