import numpy as np
from utils.mathpackage import find_area

class Triangle:
    def __init__(self, pt1, pt2, pt3):
        self.p1 = np.array(pt1, dtype=float)
        self.p2 = np.array(pt2, dtype=float)
        self.p3 = np.array(pt3, dtype=float)
        self.area = find_area(self.p1, self.p2, self.p3)

    def find_u(self, p):
        return find_area(p, self.p2, self.p3) / self.area

    def find_v(self, p):
        return find_area(p, self.p1, self.p3) / self.area

    def find_w(self, p):
        return find_area(p, self.p1, self.p2) / self.area

    def barycentric_coords(self, p):
        u = self.find_u(p)
        v = self.find_v(p)
        w = 1 - u - v
        return u, v, w

    def in_triangle(self, p):
        u, v, w = self.barycentric_coords(p)
        return (u >= 0) and (v >= 0) and (w >= 0)

    def closest_point_on_edge(self, p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab

    def closest_point(self, p):
        p = np.array(p, dtype=float)
        if self.in_triangle(p):
            return p

        u, v, w = self.barycentric_coords(p)
        if u < 0:
            return self.closest_point_on_edge(p, self.p2, self.p3)
        elif v < 0:
            return self.closest_point_on_edge(p, self.p1, self.p3)
        elif w < 0:
            return self.closest_point_on_edge(p, self.p1, self.p2)
        else:
            raise RuntimeError("Unexpected geometry condition")
