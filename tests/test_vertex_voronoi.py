from slam import generate_parametric_surfaces as sps
from slam import vertex_voronoi as svv

test_mesh = sps.generate_sphere_random_sampling(10)
acceptable_error = 0.1


def test_vertex_voronoi():
    """
    compare vertex_voronoi.sum() with mesh.area
    :return:
    """

    vert_vor = svv.vertex_voronoi(test_mesh)
    error = vert_vor.sum() - test_mesh.area
    perc_error = 100 * error / test_mesh.area
    assert perc_error < acceptable_error
