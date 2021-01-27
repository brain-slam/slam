import itertools
import random
import unittest

import numpy as np
import trimesh
from scipy.spatial import Delaunay

import slam.io as sio
import slam.topology as stop

# UTILITIES


def create_test_graph():
    """
    create a toy graph for testing
    """
    # (2)--(3)
    #  |\   |
    #  | \  |
    #  |  \ |
    #  |   \|
    # (0)--(1)

    coords = []
    coords.append([0, 0, 0])
    coords.append([0, 1, 0])
    coords.append([1, 0, 0])
    coords.append([1, 1, 0])
    coords = np.array(coords)
    faces = []
    faces.append([0, 1, 2])
    faces.append([1, 3, 2])
    faces = np.array(faces)

    return trimesh.Trimesh(faces=faces, vertices=coords, process=False)


def distinct_triangles(n=3):
    """ Create n distinct triangles """
    coords = []
    for i in range(n):
        coords.append([i, 1, 0])
        coords.append([i, 0, 0])
        coords.append([i, 0, 1])
    coords = np.array(coords)

    faces = []
    for i in range(n):
        faces.append([i * 3, i * 3 + 1, i * 3 + 2])
    faces = np.array(faces)

    return trimesh.Trimesh(faces=faces, vertices=coords, process=False)


def random_triangles(n=3):
    """ Create n triangles which can overlap """

    coords = []
    for i in range(n * 3):
        coords.append([random.random() * 5, random.random()
                       * 5, random.random() * 5])
    coords = np.array(coords)

    faces = []
    for i in range(n):
        faces.append([i * 3, i * 3 + 1, i * 3 + 2])
    faces = np.array(faces)

    return trimesh.Trimesh(faces=faces, vertices=coords, process=False)


def make_cut_sphere_a():
    """ Create a sphere cut by two parallel planes """
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    mesh_a = trimesh.intersections.slice_mesh_plane(
        mesh_a, [
            0, 1, 0], [
            0, -.1, 0])
    mesh_a = trimesh.intersections.slice_mesh_plane(
        mesh_a, [0, -1, 0], [0, .9, 0])
    return mesh_a


def make_cut_sphere_b():
    """ Create a sphere cut by one parallel plane """
    mesh_a = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    mesh_a = trimesh.intersections.slice_mesh_plane(
        mesh_a, [
            0, 1, 0], [
            0, -.1, 0])
    return mesh_a


def create_K_rings(K=4):
    """ Starts from a regular hexagon and add several k-ring"""

    coords = [[0.0, 0.0]]
    # k-ring, k=1-2, can be generalized to more than k
    for k in range(1, K):
        angle = (np.pi / 3) / k
        for i in range(6 * k):
            coords.append([k * np.cos(i * angle), k * np.sin(i * angle)])

    coords = np.array(coords)
    print(coords)

    tri = Delaunay(coords)

    return trimesh.Trimesh(faces=tri.simplices, vertices=coords, process=False)


class TestTopologyMethods(unittest.TestCase):

    # Sphere cut by two planes, with new vertices added at the intersection
    cutSphere_A = make_cut_sphere_a()

    # Sphere cut by one plane, with new vertices added at the intersection
    cutSphere_B = make_cut_sphere_b()

    # Sphere cut by two planes, with no new vertices added at the intersection
    cutSphere_C = sio.load_mesh("tests/data/topology/mesh_C.gii")

    # A 3D disk with height 0 and radius 2
    disk_radius_2 = sio.load_mesh("tests/data/topology/mesh_D.gii")

    # hexagon 2 rings
    K = 4
    k_rings = create_K_rings(K)

    def test_boundary_angles(self):
        # for a k ring the angle between two consecutive sides is pi - pi/(3k)
        N = len(self.k_rings.vertices)
        coords = np.hstack([self.k_rings.vertices, np.zeros((N, 1))])
        indices = range(N-6*(self.K-1), N)
        ang, norm = stop.boundary_angles(indices, coords)
        assert(ang == 180 - 180/(3 * (self.K - 1))).all

    def test_boundaries_basic(self):
        mesh_a = self.cutSphere_A

        mesh_a_save = mesh_a.copy()

        boundary = stop.mesh_boundary(mesh_a)

        # Non modification
        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        assert(mesh_a.faces == mesh_a_save.faces).all()

        # Correctness
        assert(len(boundary) == 2)

        # Type
        assert(isinstance(boundary, list))
        assert(isinstance(boundary[0], list))

        iterable = list(itertools.combinations(boundary, 2))

        # Uniqueness
        for b1, b2 in iterable:
            assert set(b1).intersection(set(b2)) == set()

    def test_boundaries_triangles(self):

        mesh_a = distinct_triangles(5)
        mesh_b = random_triangles(5)

        boundary_a = stop.mesh_boundary(mesh_a)
        boundary_b = stop.mesh_boundary(mesh_b)

        iterable_a = list(itertools.combinations(boundary_a, 2))
        iterable_b = list(itertools.combinations(boundary_b, 2))

        # Correctness
        assert(len(boundary_a) == 5)
        assert(len(boundary_b) == 5)

        # Uniqueness
        for b1, b2 in iterable_a:
            assert set(b1).intersection(set(b2)) == set()
        for b1, b2 in iterable_b:
            assert set(b1).intersection(set(b2)) == set()

    def test_angle_norm_boundary(self):

        # GET A DISK IN 3D

        radius = 2
        mesh_a = self.disk_radius_2
        coords3D = mesh_a.vertices

        # COMPUTE BOUNDARY

        boundary = stop.mesh_boundary(mesh_a)

        # Correctness
        assert(len(boundary) == 1)

        # COMPUTE ANGLE AND NORM VARIATIONS

        angles, norms = stop.boundary_angles(boundary[0], coords3D)
        angles = abs(180 - angles)

        sum_angles, sum_norms = sum(angles), sum(norms)

        # Computation on three arbitrary linked vertices
        three_angles, three_norms = stop.boundary_angles([0, 1, 2], coords3D)
        three_angles = abs(180 - three_angles)

        perimeter = radius * np.pi * 2
        precision_A = .001
        precision_B = .001

        # Coherence of the angle sum and norm sum
        assert(np.isclose(sum_angles, 360, precision_A))
        assert(np.isclose(sum_norms, perimeter, precision_B))

        # Coherence of a partial angle and a partial norm variation
        assert(np.isclose(three_angles[1], 360 / len(coords3D), precision_A))
        assert(
            np.isclose(
                three_norms[1],
                perimeter /
                len(coords3D),
                precision_B))

    def test_boundaries_intersection_copy(self):
        mesh_a = self.cutSphere_A

        boundary = stop.mesh_boundary(mesh_a)
        b1 = boundary[0]
        b2 = boundary[1]
        b2_save = b2.copy()

        concat = b1 + b2

        inter_1 = stop.boundaries_intersection([concat, b2])
        inter_1 = inter_1[0][2]

        # Non modification
        assert(b2 == b2_save)

        # Type
        assert (isinstance(inter_1, list))

        # Correctness
        assert(set(inter_1) == set(b2))

    def test_close_mesh(self):
        mesh_a = self.cutSphere_C.copy()

        boundary_prev = stop.mesh_boundary(mesh_a)

        # Correctness
        assert(len(boundary_prev) == 2)

        mesh_a_closed, nvertices_added = stop.close_mesh(mesh_a)

        # Coherence
        assert(len(mesh_a_closed.vertices) == len(
            mesh_a.vertices) + nvertices_added)

        boundary_closed = stop.mesh_boundary(mesh_a_closed)

        # Mesh is now closed
        assert(len(boundary_closed) == 0)

        # Non modification of the vertices which are not on the boundaries
        sbase = (set(range(len(mesh_a.vertices))))
        sbound1 = (set(boundary_prev[0]))
        sbound2 = (set(boundary_prev[1]))
        sbound = sbound1.union(sbound2)
        snot_bound = (sbase.difference(sbound))

        for i in snot_bound:
            assert (mesh_a.vertices[i] == mesh_a_closed.vertices[i]).all()

    def test_remove_mesh_boundary_faces(self):
        mesh_a = self.cutSphere_A.copy()
        mesh_a_save = mesh_a.copy()
        boundary = stop.mesh_boundary(mesh_a)
        mesh_processed = stop.remove_mesh_boundary_faces(mesh_a,
                                                         face_vertex_number=1)
        # Non modification
        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        # Correctness
        # check that when removing all faces with any vertex on the boundary,
        # all the boundary vertices are removed from the mesh
        print(len(boundary[0]))
        print(len(mesh_a.vertices))
        print(len(mesh_processed.vertices))
        assert(len(mesh_processed.vertices) ==
               len(mesh_a.vertices) - len(boundary[0]))

    def test_k_ring_neighborhood(self):
        mesh_hexagon = self.k_rings.copy()
        # WARNING, 0 is not the center...
        texture_2_rings = stop.k_ring_neighborhood(mesh_hexagon, index=0, k=2)
        zero_ring = np.where(texture_2_rings == 0)[0]
        one_ring = np.where(texture_2_rings == 1)[0]
        two_ring = np.where(texture_2_rings == 2)[0]
        assert (zero_ring == [0]).all()
        assert (one_ring == [i for i in range(1, 7)]).all()
        assert (two_ring == [i for i in range(7, 19)]).all()

    def test_adjacency_matrix(self):
        toy_graph = create_test_graph()
        gd_truth_adja = []
        gd_truth_adja.append([0, 1, 1, 0])
        gd_truth_adja.append([1, 0, 1, 1])
        gd_truth_adja.append([1, 1, 0, 1])
        gd_truth_adja.append([0, 1, 1, 0])
        gd_truth_adja = np.array(gd_truth_adja)
        adja = stop.adjacency_matrix(toy_graph)
        assert (adja == gd_truth_adja).all()


if __name__ == '__main__':
    unittest.main()
