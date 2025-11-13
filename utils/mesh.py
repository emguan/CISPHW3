"""
Custom class for Mesh object and all corresponding helper functions.

Author: Emily Guan
"""

from typing import List, Tuple
import numpy as np
from utils.triangles import Triangle

class Mesh:
    def __init__(self, vertices, indices):
        self.triangles: List[Triangle] = [] # list of triangles
        self.build_mesh(vertices, indices)

    """
    Loops through all input vertices + indices and gives a list of Triangles. 
    """
    def build_mesh(self, vertices, indices):
        vertices = [np.array(v, dtype=float) for v in vertices]

        for (i1, i2, i3) in indices:
            tri = Triangle(vertices[i1], vertices[i2], vertices[i3])
            self.triangles.append(tri)

    def __len__(self):
        return len(self.triangles)

    def __getitem__(self, idx):
        return self.triangles[idx]

    """
    Linear search for closest point on a triangle to given point. 
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
    """
    def find_closest_point(self, p):
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
