import numpy as np
import math

class Triangle:
    def __init__(self, pt1, pt2, pt3):
        self.a = np.array(pt1, dtype=float)
        self.b = np.array(pt2, dtype=float)
        self.c = np.array(pt3, dtype=float)

        self.normal = np.cross(self.b - self.a, self.c - self.a)
        self.normal = self.normal / np.linalg.norm(self.normal)

    def barycentric_coords(self, p):
        v0 = self.b - self.a
        v1 = self.c - self.a
        v2 = p - self.a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d10 = np.dot(v1, v0)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    def project_to_plane(self, p):
        p = np.array(p, dtype=float)
        dist = np.dot(p - self.a, self.normal)
        return p - dist * self.normal

    def closest_point_on_edge(self, p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab

    def closest_point(self, p):
        p = np.array(p, dtype=float)

        # 1. Project p onto triangle plane
        p_proj = self.project_to_plane(p)

        # 2. Barycentric of projection
        u, v, w = self.barycentric_coords(p_proj)

        # 3. If inside: return projection
        if (u >= 0) and (v >= 0) and (w >= 0):
            return p_proj

        # 4. Outside: find edge closest point using 3D edges
        c1 = self.closest_point_on_edge(p, self.a, self.b)
        c2 = self.closest_point_on_edge(p, self.b, self.c)
        c3 = self.closest_point_on_edge(p, self.c, self.a)

        # 5. Choose point closest to original p (not p_proj!)
        candidates = [c1, c2, c3]
        dists = [np.linalg.norm(p - c) for c in candidates]

        return candidates[np.argmin(dists)]
