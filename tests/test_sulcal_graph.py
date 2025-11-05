import unittest
import numpy as np
import networkx as nx
from slam import sulcal_graph, geodesics
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

        g = sulcal_graph.add_node_attribute_to_graph(self.graph, a, name='test', save=False)

        self.assertIn("test", g.nodes[0].keys())
        self.assertEqual(a[0], g.nodes[0]['test'])
        self.assertTrue(
            (a[list(nx.get_node_attributes(g, 'pit_index'))] == list(nx.get_node_attributes(g, 'test'))).all())

    def test_add_edge_attribute(self):
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        g = sulcal_graph.add_edge_attribute_to_graph(self.graph, a, name='test', save=False)

        self.assertIn("test", g.edges[list(g.edges)[0]].keys())
        self.assertEqual(a[4], g.edges[list(g.edges)[0]]['test'])
        self.assertTrue(
            (a[list(nx.get_edge_attributes(g, 'ridge_index'))] == list(nx.get_edge_attributes(g, 'test'))).all())

    def test_add_mean_value_to_graph(self):
        g = sulcal_graph.add_mean_value_to_graph(self.graph, self.texture, name="mean_texture", save=False)

        self.assertIn("mean_texture", g.nodes[0].keys())
        self.assertAlmostEqual(g.nodes[0]["mean_texture"], 0.15)
        self.assertAlmostEqual(g.nodes[1]["mean_texture"], 0.35)

    def test_get_textures_from_graph(self):
        # Test de la fonction get_textures_from_graph
        tex_labels, tex_pits, tex_ridges = sulcal_graph.get_textures_from_graph(self.graph, self.mesh, save=False)

        self.assertIsInstance(tex_labels, texture.TextureND)
        self.assertIsInstance(tex_pits, texture.TextureND)
        self.assertIsInstance(tex_ridges, texture.TextureND)
        self.assertEqual(tex_labels.darray.shape[1], len(self.mesh.vertices))
        self.assertEqual(tex_pits.darray.shape[1], len(self.mesh.vertices))
        self.assertEqual(tex_ridges.darray.shape[1], len(self.mesh.vertices))

    def test_add_geodesic_distances(self):
        def mock_compute_gdist(mesh, ridge):
            return np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        geodesics.compute_gdist = mock_compute_gdist
        g = sulcal_graph.add_geodesic_distances_to_graph(self.graph, self.mesh, save=False)
        print(g.edges[list(g.edges)[0]])

        self.assertIn("geodesic_distance_btw_ridge_pit_i", g.edges[list(g.edges)[0]].keys())
        self.assertIn("geodesic_distance_btw_ridge_pit_j", g.edges[list(g.edges)[0]].keys())
        self.assertIn("geodesic_distance_btw_pits", g.edges[list(g.edges)[0]].keys())


if __name__ == '__main__':
    unittest.main()