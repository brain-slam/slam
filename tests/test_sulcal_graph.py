import unittest
from unittest.mock import patch
import numpy as np
import networkx as nx
from slam import sulcal_graph
from slam import texture


class TestSulcalGraph(unittest.TestCase):

    def setUp(self):
        # Create a synthetic graph
        self.graph = nx.Graph()
        self.graph.add_node(0, basin_vertices=[0, 1], pit_index=0, basin_label=0)
        self.graph.add_node(1, basin_vertices=[2, 3], pit_index=2, basin_label=1)
        self.graph.add_edge(0, 1, ridge_index=4)

        # Create a synthetic mesh
        class MockMesh:
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]])

        self.mesh = MockMesh()

        # Create a synthetic texture
        self.texture = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    def test_add_node_attribute(self):
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        g = sulcal_graph.add_node_attribute_from_texture(self.graph, a, attribute_name='test')

        self.assertIn("test", g.nodes[0].keys())
        self.assertEqual(a[0], g.nodes[0]['test'])
        self.assertTrue(
            (a[list(nx.get_node_attributes(g, 'pit_index'))] == list(nx.get_node_attributes(g, 'test'))).all())

    def test_add_edge_attribute(self):
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        g = sulcal_graph.add_edge_attribute_from_texture(self.graph, a, attribute_name='test')

        self.assertIn("test", g.edges[list(g.edges)[0]].keys())
        self.assertEqual(a[4], g.edges[list(g.edges)[0]]['test'])
        self.assertTrue(
            (a[list(nx.get_edge_attributes(g, 'ridge_index'))] == list(nx.get_edge_attributes(g, 'test'))).all())

    def test_add_mean_value_to_graph(self):
        g = sulcal_graph.add_mean_value_to_nodes(self.graph, self.texture, attribute_name="mean_texture")

        self.assertIn("mean_texture", g.nodes[0].keys())
        self.assertAlmostEqual(g.nodes[0]["mean_texture"], 0.15)
        self.assertAlmostEqual(g.nodes[1]["mean_texture"], 0.35)

    def test_get_textures_from_graph(self):
        # Test de la fonction get_textures_from_graph
        atex_labels, atex_pits, atex_ridges = sulcal_graph.get_textures_from_graph(self.graph, self.mesh)

        self.assertIsInstance(atex_labels, np.ndarray)
        self.assertIsInstance(atex_pits, np.ndarray)
        self.assertIsInstance(atex_ridges, np.ndarray)
        self.assertEqual(atex_labels.shape[0], len(self.mesh.vertices))
        self.assertEqual(atex_pits.shape[0], len(self.mesh.vertices))
        self.assertEqual(atex_ridges.shape[0], len(self.mesh.vertices))

    def test_add_geodesic_distances(self):
        def mock_compute_gdist(mesh, ridge):
            return np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Use unittest.mock.patch as a context manager (with) to ensures that the original function compute_gdist
        # is automatically restored after the block ends.
        with patch('slam.geodesics.compute_gdist', side_effect=mock_compute_gdist):
            g = sulcal_graph.add_geodesic_distances_to_edges(self.graph, self.mesh)
            print(g.edges[list(g.edges)[0]])

            self.assertIn("geodesic_distance_btw_ridge_pit_i", g.edges[list(g.edges)[0]].keys())
            self.assertIn("geodesic_distance_btw_ridge_pit_j", g.edges[list(g.edges)[0]].keys())
            self.assertIn("geodesic_distance_btw_pits", g.edges[list(g.edges)[0]].keys())


if __name__ == '__main__':
    unittest.main()