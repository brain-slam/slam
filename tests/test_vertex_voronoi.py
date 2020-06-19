from slam import generate_parametric_surfaces as sps
from slam import vertex_voronoi as svv
import unittest



class TestVertexVoronoiMethods(unittest.TestCase):

    def test_vertex_voronoi(self):
        """
        compare vertex_voronoi.sum() with mesh.area
        :return:
        """
        mesh_A = sps.generate_sphere_random_sampling(10)
        acceptable_error = 0.000001
        vert_vor = svv.vertex_voronoi(mesh_A)
        assert vert_vor.sum() - mesh_A.area < acceptable_error


if __name__ == '__main__':
    unittest.main()