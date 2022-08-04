import unittest

import numpy as np
from scipy.spatial import Delaunay
import trimesh
import slam.geodesics as sg

TOL= 1e-10

def create_rectangular_grid(rows=3, cols=4):
    """
    Create a rectangular graph
    :param rows:
    :param cols:
    :return:
    """
    x = np.arange(0, rows, 1)
    y = np.arange(0, cols, 1)

    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = np.zeros((len(X),))

    coords = np.vstack((X, Y, Z)).T
    tri = Delaunay(np.vstack((X, Y)).T, qhull_options="QJ Qt Qbb")
    print(tri)

    return trimesh.Trimesh(faces=tri.simplices, vertices=coords, process=False)


class TestGeodesics(unittest.TestCase):

    def test_compute_gdist(self):
        # Check if the distance between two points of a diagonal is bounded between L2 norm and L1 norm
        m = 4
        n = 5
        rectangle = create_rectangular_grid(m, n)
        print(rectangle)
        distance = sg.compute_gdist(rectangle, 0)
        low_bound = np.sqrt((m-1)**2 + (n-1)**2)  # L2 diameter
        up_bound = m-1 + n-1  # L1 diameter
        self.assertTrue(np.logical_and(distance[m*n-1] >= low_bound, distance[-1] <= up_bound))

    def test_dijkstra_length(self):
        # Check if the max distance to one side of the rectangle equals the length of the other side
        m = 4
        n = 5
        rectangle = create_rectangular_grid(m, n)
        set_of_points = range(m)  # boundary with y = 0
        distance = sg.dijkstra_length(rectangle, set_of_points)
        self.assertTrue(np.abs(np.max(distance) - (n-1) < TOL))


if __name__ == "__main__":
    unittest.main()