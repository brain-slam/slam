import unittest

import trimesh


def make_sphere(radius=1):
    """ Create a sphere"""
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    return mesh_a


class TestSurfaceProfiling(unittest.TestCase):
    def test_basic(self):
        print(1)
        pass


if __name__ == "__main__":
    unittest.main()
