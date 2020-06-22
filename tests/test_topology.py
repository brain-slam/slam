import slam.io as sio
import slam.topology as stop
import slam.plot as splt
import slam.generate_parametric_surfaces as sps
import numpy as np
from vispy.scene import Line
from visbrain.objects import VispyObj, SourceObj
import sys
import unittest
import trimesh
import itertools
import random

### UTILITIES

def distinct_triangles(n=3):
    """ Draw n distinct triangles """ 
    coords = []
    for i in range(n):
        coords.append([i, 1,0])
        coords.append([i, 0,0])
        coords.append([i, 0,1])
    coords = np.array(coords)

    faces = []
    for i in range(n):
        faces.append([i*3, i*3+1, i*3+2])
    faces = np.array(faces)

    return trimesh.Trimesh(faces=faces, vertices=coords, process=False)
    
def random_triangles(n=3):
    """ Draw n triangles which can overlap """ 

    coords = []
    for i in range(n*3):
        coords.append([random.random()*5, random.random()*5,random.random()*5])
    coords = np.array(coords)

    faces = []
    for i in range(n):
        faces.append([i*3, i*3+1, i*3+2])
    faces = np.array(faces)
    
    return trimesh.Trimesh(faces=faces, vertices=coords, process=False)


def make_cutSphere_A():
    mesh_a =  trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    mesh_a = trimesh.intersections.slice_mesh_plane(mesh_a,[0,1,0], [0,-.1,0])
    mesh_a = trimesh.intersections.slice_mesh_plane(mesh_a,[0,-1,0], [0,.9,0])
    return mesh_a

def make_cutSphere_B():
    mesh_a =  trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    mesh_a = trimesh.intersections.slice_mesh_plane(mesh_a,[0,1,0], [0,-.1,0])
    mesh_a = trimesh.intersections.slice_mesh_plane(mesh_a,[0,-1,0], [0,.9,0])
    return mesh_a


class TestTopologyMethods(unittest.TestCase):


    cutSphere_A = make_cutSphere_A()
    cutSphere_B = make_cutSphere_B()


    def test_boundaries_basic(self):
        mesh_a =  self.cutSphere_A

        mesh_a_save = mesh_a.copy()

        boundary = stop.mesh_boundary(mesh_a)


        # Non modification
        assert(mesh_a.vertices == mesh_a_save.vertices).all()
        assert(mesh_a.faces == mesh_a_save.faces).all()

        # Correctness
        assert(len(boundary) == 2)

        # Type
        assert(type(boundary) == list)
        assert(type(boundary[0]) == list)

        iterable = list(itertools.combinations(boundary, 2))
        
        # Uniqueness
        for b1,b2 in iterable:
            assert set(b1).intersection(set(b2)) == set()

    
    def test_boundaries_triangles(self):

        mesh_a =  distinct_triangles(5)
        mesh_b =  random_triangles(5)

        open_mesh_boundary_a = stop.mesh_boundary(mesh_a)
        open_mesh_boundary_b = stop.mesh_boundary(mesh_b)

        iterable_a = list(itertools.combinations(open_mesh_boundary_a, 2))
        iterable_b = list(itertools.combinations(open_mesh_boundary_b, 2))
        
        # Uniqueness
        for b1,b2 in iterable_a:
            assert set(b1).intersection(set(b2)) == set()
        for b1,b2 in iterable_b:
            assert set(b1).intersection(set(b2)) == set()


    def test_angle_norm_boundary(self):

        ### CREATE A DISK IN 3D

        radius = 2
        mesh_a = trimesh.path.creation.circle(radius)

        coords2D,faces = mesh_a.triangulate()
        coords3D = []
        for i,v in enumerate(coords2D):
            coords3D.append([v[0],v[1],0])

        coords3D = np.array(coords3D)

        mesh_a = trimesh.Trimesh(faces=faces, vertices=coords3D, process=False)

        ### COMPUTE BOUNDARY

        boundary = stop.mesh_boundary(mesh_a)

        assert(len(boundary) == 1)

        ### COMPUTE ANGLES AND NORM VARIATIONS

        angles,norms = stop.boundary_angles(boundary[0], coords3D)
        angles = abs(180-angles)
        
        sum_angles, sum_norms = sum(angles), sum(norms)

        three_angles,three_norms = stop.boundary_angles([0,1,2], coords3D)
        three_angles = abs(180-three_angles)

        perimeter = radius*np.pi*2
        precision_A = .001
        precision_B = .001

        assert(np.isclose(three_norms[1], perimeter/len(coords3D), precision_A))
        assert(np.isclose(three_angles[1], 360/len(coords3D), precision_B))

        assert(np.isclose(sum_angles, 360, precision_A))
        assert(np.isclose(sum_norms, perimeter, precision_B))


    def test_boundaries_concat_basic(self):
        mesh_a =  self.cutSphere_A

        boundary = stop.mesh_boundary(mesh_a)

        assert (type(boundary) == list)


    def test_boundaries_concat_copy(self):
        mesh_a =  self.cutSphere_A

        boundary = stop.mesh_boundary(mesh_a)
        b1_save = boundary[0].copy()
        b1 = boundary[0]
        b2 = boundary[1]

        concat = stop.cat_boundary(b1, b2)

        # Non modification
        # assert(len(b1) == len(b1_save))
        # assert((b1==b1_save))

        inter_1 = stop.boundaries_intersection([concat, b2])
        inter_1 = inter_1[0][2]

        assert(set(inter_1) == set(b2))

if __name__ == '__main__':
    unittest.main()