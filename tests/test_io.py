import unittest
import tempfile
import numpy as np
import trimesh

import slam.io as sio

# UTILITIES


def make_sphere_a():
    """Create a sphere"""
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    return mesh_a


class TestIOMethods(unittest.TestCase):

    # Sphere
    sphere_A = make_sphere_a()

    def test_basic(self):
        mesh_a = self.sphere_A.copy()
        mesh_a_save = self.sphere_A.copy()

        fo = tempfile.NamedTemporaryFile(suffix=".gii")

        sio.write_mesh(mesh_a, fo.name)

        # Non modification
        assert (mesh_a.vertices == mesh_a_save.vertices).all()
        assert (mesh_a.faces == mesh_a_save.faces).all()

        mesh_b = sio.load_mesh(fo.name)

        fo.close()

        precision_A = 0.00001

        # Correctness
        assert np.isclose(mesh_a.vertices, mesh_b.vertices, precision_A).all()
        assert np.isclose(mesh_a.faces, mesh_b.faces, precision_A).all()

    def test_load_texture(self):
        test_text = sio.load_texture('examples\data\example_texture_parcel.gii').darray[0]
        self.assertTrue(test_text.size>0)


if __name__ == "__main__":
    unittest.main()
