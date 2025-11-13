"""
Custom class for Triangle object and all corresponding helper functions.

Author: Emily Guan
"""

import numpy as np
import math

class Triangle:

    """
    Input:
        pt1, pt2, pt3: array-like (3,)
            Coordinates of the triangle's vertices.

    Output:
        Constructs a Triangle with computed:
            - bounding box (lb, ub)
            - unit normal vector
    """
    def __init__(self, pt1, pt2, pt3):
        self.a = np.array(pt1, dtype=float)
        self.b = np.array(pt2, dtype=float)
        self.c = np.array(pt3, dtype=float)

        self.lb, self.ub = self.build_bounds() 
        self.normal = self.compute_normal()

    """
    Computes unit vector of triangle. 

    Output:
        normal: numpy array (3,)
            Unit vector perpendicular to triangle plane.
    """
    def compute_normal(self):
        normal = np.cross(self.b - self.a, self.c - self.a)
        normal = normal / np.linalg.norm(normal)
        return normal
    
    """
    Builds bounds for later use during box bounding. 

    Output:
        lb: numpy array (3,)
            Lower bounds (min x, min y, min z)

        ub: numpy array (3,)
            Upper bounds (max x, max y, max z)
    """
    def build_bounds(self):
        xs = (self.a[0], self.b[0], self.c[0])
        ys = (self.a[1], self.b[1], self.c[1])
        zs = (self.a[2], self.b[2], self.c[2])
        
        lb = np.array([min(xs), min(ys), min(zs)], dtype=float)
        ub = np.array([max(xs), max(ys), max(zs)], dtype=float)

        return lb, ub

    """
    Build barycentric coordinates.

    Originally used area formula approach. 
    Currently using vertex approach from https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates 
    
    Input:
        p: 3D point to convert.

    Output:
        u, v, w: floats
            Barycentric coordinates satisfying u + v + w = 1.
            If u, v, w >= 0 → p_proj lies inside the triangle.
    """
    def barycentric_coords(self, p):
        v0 = self.b - self.a
        v1 = self.c - self.a
        v2 = p - self.a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    """
    Projects point (modeled as vector) onto base of triangle.

    Uses dot product for projection. 

    Input:
        p: Point in space.

    Output:
        p_proj: numpy array (3,)
            Orthogonal projection of p onto triangle plane.
    """
    def project_to_plane(self, p):
        p = np.array(p, dtype=float)
        dist = np.dot(p - self.a, self.normal)
        return p - dist * self.normal
    
    """
    Checks if a point is within a certain bounding box 
    (defined as bounding box of triangle + some bound.)

    Input:
        p: Query point.

        margin: float
            Expansion distance around bounding box.

    Output:
        True if p lies within [lb - margin, ub + margin], otherwise False.
        
    """
    def in_box(self, p, margin):
        return np.all(p >= self.lb - margin) and np.all(p <= self.ub + margin)

    """
    If closest point is an edge, finds which point.

    Input:
        p: Query point.

        a, b: numpy arrays (3,)
            Endpoints of the segment.

    Output:
        c: numpy array (3,)
            Closest point on the segment AB.
    """
    def closest_point_on_edge(self, p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab

    """
    Finds closest point on a triangle. 

    Input:
        p: Query point.

    Output:
        closest: numpy array (3,)
            Closest point on the triangle.
    """
    def closest_point(self, p):
        p = np.array(p, dtype=float)

        # project p onto triangle
        p_proj = self.project_to_plane(p)

        # barycentric coords
        u, v, w = self.barycentric_coords(p_proj)

        # if inside, return proejction
        if (u >= 0) and (v >= 0) and (w >= 0):
            return p_proj

        # if outside, find edge closest point on all edges
        c1 = self.closest_point_on_edge(p, self.a, self.b)
        c2 = self.closest_point_on_edge(p, self.b, self.c)
        c3 = self.closest_point_on_edge(p, self.c, self.a)

        # choose projection closest to original p
        candidates = [c1, c2, c3]
        dists = [np.linalg.norm(p - c) for c in candidates]

        return candidates[np.argmin(dists)]
