import slam.io as sio
import numpy as np
import unittest
import trimesh

# UTILITIES


def make_sphere_a():
    """ Create a sphere"""
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return mesh_a


class TestIOMethods(unittest.TestCase):

    # Sphere
    sphere_A = make_sphere_a()

    def test_basic(self):
        mesh_a = self.sphere_A.copy()
        mesh_a_save = self.sphere_A.copy()

        sio.write_mesh(mesh_a, "tests/data/io/temp.gii")

        # Non modification
        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        assert(mesh_a.faces == mesh_a_save.faces).all()

        mesh_b = sio.load_mesh('tests/data/io/temp.gii')

        precision_A = .00001

        # Correctness
        assert(np.isclose(mesh_a.vertices, mesh_b.vertices, precision_A).all())
        assert(np.isclose(mesh_a.faces, mesh_b.faces, precision_A).all())


if __name__ == '__main__':
    unittest.main()
