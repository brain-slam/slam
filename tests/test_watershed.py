import unittest
import numpy as np
import slam.watershed as swat
import trimesh


def make_sphere():
    """Create a sphere"""
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return mesh_a


class TestWatershed(unittest.TestCase):

    def setUp(self):

        # Create a sphere
        self.sphere = make_sphere()

        # Create a simpler synthetic mesh
        self.mesh = self._create_mock_mesh()
        self.dpf = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.voronoi = np.array([1, 1, 1, 1, 1])
        self.vert_neigh = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
        self.thresh_dist = 0.3
        self.thresh_area = 2
        self.thresh_ridge = 1

    def _create_mock_mesh(self):
        class MockMesh:
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]])

        return MockMesh()

    def test_watershed(self):
        _, dpf, voronoi = swat.compute_mesh_features(self.sphere)

        basins, ridges, adjacency = swat.watershed(
            self.sphere, voronoi, dpf, self.thresh_dist, self.thresh_ridge, self.thresh_area
        )

        self.assertIsInstance(basins, dict)
        self.assertIsInstance(ridges, dict)
        self.assertIsInstance(adjacency, np.ndarray)
        self.assertGreater(len(basins), 0, "At least one basin should be detected.")
        # self.assertEqual(adjacency.shape[0], adjacency.shape[1], "Adjacency matrix should be square.")
        # self.assertTrue(np.array_equal(adjacency, adjacency.T), "Adjacency matrix should be symmetric.")
        basins_with_ridge = []
        [[basins_with_ridge.append(b) for b in ridge_tuple] for ridge_tuple in ridges]
        self.assertTrue((np.sort(np.unique(basins_with_ridge)) == list(basins.keys())).all(), "Ridges and basins "
                                                                                              "dictionaries have "
                                                                                              "different basins labels")

    def test_get_textures_from_dict(self):
        basins = {
            0: {"basin_vertices": [0, 1], "pit_index": 0},
            1: {"basin_vertices": [2, 3], "pit_index": 2},
        }
        ridges = {
            (0, 1): {"ridge_index": 4, "ridge_depth": 0.5, "ridge_length": 1}
        }

        atex_labels, atex_pits, atex_ridges = swat.get_textures_from_dict(
            self.mesh, basins, ridges
        )

        self.assertIsInstance(atex_labels, np.ndarray)
        self.assertIsInstance(atex_pits, np.ndarray)
        self.assertIsInstance(atex_ridges, np.ndarray)
        self.assertEqual(atex_labels.shape[0], len(self.mesh.vertices))
        self.assertEqual(atex_pits.shape[0], len(self.mesh.vertices))
        self.assertEqual(atex_ridges.shape[0], len(self.mesh.vertices))


if __name__ == '__main__':
    unittest.main()