"""
Custom class for Mesh object and all corresponding helper functions.

Author: Emily Guan
"""

from typing import List, Tuple
import numpy as np
from utils.triangles import Triangle

class Mesh:
    """
    Input:
        vertices: array-like (N_vertices x 3)
            3D vertex coordinates from the mesh file.

        indices: array-like (N_triangles x 3)
            Indices defining a triangle.

    Output:
        Initializes the Mesh object by building a list of Triangle objects.
    """
    def __init__(self, vertices, indices):
        self.triangles: List[Triangle] = [] # list of triangles
        self.build_mesh(vertices, indices)

    """
    Loops through all input vertices + indices and gives a list of Triangles. 
    
    Input:
        vertices: list/array of shape (N_vertices, 3)
            Vertex positions.

        indices: list/array of shape (N_triangles, 3)
            Triples of integer indices referring to vertex list rows.

    Returns:
        self.triangles with Triangle objects.
    """
    def build_mesh(self, vertices, indices):
        vertices = [np.array(v, dtype=float) for v in vertices]

        for (i1, i2, i3) in indices:
            tri = Triangle(vertices[i1], vertices[i2], vertices[i3])
            self.triangles.append(tri)

    """
    Output:
        Returns the number of triangles in the mesh.
    """
    def __len__(self):
        return len(self.triangles)

    """
    Input:
        idx: int
            Index of the triangle.

    Output:
        Returns the Triangle at the given index.
    """
    def __getitem__(self, idx):
        return self.triangles[idx]

    """
    Linear search for closest point on a triangle to given point. 
    
    Input:
        p: Query point in 3D space.

    Output:
        closest_point: numpy array (3,)
            Closest point on mesh surface to p.
    """
    def find_closest_point_linear(self, p):
        p = np.array(p, dtype=float)
        closest_point = None
        min_dist = float('inf')

        # choose candiate with lowest distance
        for tri in self.triangles:
            candidate = tri.closest_point(p)
            dist = np.linalg.norm(candidate - p)
            if dist < min_dist:
                min_dist = dist
                closest_point = candidate
        
        return closest_point
    
    """
    Bounded box search for closest point on a triangle to given point. 
    
    Input:
        p: Query point.

    Output:
        closest_point: numpy array (3,)
            Closest point on the surface.
    """
    def find_closest_point_box(self, p):
        p = np.array(p, dtype=float)
        closest_point = None
        bound = float('inf')

        # random starting point
        for tri in self.triangles[:6]:
            cp = tri.closest_point(p)
            d = np.linalg.norm(cp - p)
            if d < bound:
                bound = d
                closest_point = cp

        # if in a smaller box, replace
        for tri in self.triangles:
            if not tri.in_box(p, bound):
                continue

            candidate = tri.closest_point(p)
            dist = np.linalg.norm(candidate - p)

            if dist < bound:
                bound = dist
                closest_point = candidate
        
        return closest_point

    def find_closest_point(self, points, use_linear=False):

        points = np.asarray(points)

        out = np.zeros_like(points)
        for i, p in enumerate(points):
            out[i] = self.find_closest_point_box(p)

        return out