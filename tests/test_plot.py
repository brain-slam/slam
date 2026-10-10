import unittest
import numpy as np
import plotly.graph_objects as go
from slam import plot as splot


class TestMeshProjection(unittest.TestCase):

    def setUp(self):
        """Initialisation des données simulées basées sur votre exemple."""
        # Simulation de 10 sommets et de faces
        self.vertices = np.random.rand(10, 3)
        self.faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        self.mean_curv = np.random.randn(10)  # Une valeur par sommet

    def test_full_workflow(self):
        """Teste l'intégration complète avec des données d'intensité par sommet."""
        # 1. Préparation des dictionnaires (exactement comme votre exemple)
        display_settings = {'colorbar_label': 'Mean Curvature'}

        mesh_data = {
            'vertices': self.vertices,
            'faces': self.faces,
            'title': 'example_mesh.gii Mean Curvature'
        }

        intensity_data = {
            'values': self.mean_curv,
            'mode': 'vertex'
        }

        # 2. Exécution
        fig = splot.mes3d_projection(
            mesh_data=mesh_data,
            intensity_data=intensity_data,
            display_settings=display_settings
        )

        # 3. Assertions (Vérifications)
        # Vérification du type de sortie
        self.assertIsInstance(fig, go.Figure)

        # Accès à la trace Mesh3d
        mesh_trace = fig.data[0]

        # Vérification de la correspondance des données
        np.testing.assert_array_equal(mesh_trace.x, self.vertices[:, 0])
        self.assertEqual(mesh_trace.intensitymode, 'vertex')
        self.assertEqual(mesh_trace.colorbar.title.text, 'Mean Curvature')

        # Vérification que le mode 'flatshading' est activé (car intensity_data est présent)
        self.assertTrue(mesh_trace.flatshading)

    def test_missing_display_settings_keys(self):
        """Vérifie que la fonction ne plante pas si des clés sont manquantes."""
        mesh_data = {'vertices': self.vertices, 'faces': self.faces}
        # On passe un dictionnaire vide pour les réglages
        fig = splot.mes3d_projection(mesh_data, intensity_data=None, display_settings={})

        self.assertEqual(fig.data[0].color, "ghostwhite")