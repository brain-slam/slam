from slam import generate_parametric_surfaces as sps
from slam import vertex_voronoi as svv

test_mesh = sps.generate_sphere(10)
acceptable_error = 0.000001

def test_vertex_voronoi():
    """
    compare vertex_voronoi.sum() with mesh.area
    :return:
    """
    vert_vor = svv.vertexVoronoi(test_mesh)
    assert vert_vor.sum()-test_mesh.area < acceptable_error
