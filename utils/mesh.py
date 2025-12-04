from typing import List
import numpy as np
from utils.triangles import Triangle

class Mesh:
    """
    Mesh with:
      - Triangle list
      - Per-vertex normals
      - Correct interpolated normals via barycentric coords from Triangle.closest_point
    """
    def __init__(self, vertices, indices):
        self.vertices = np.asarray(vertices, float)
        self.indices = np.asarray(indices, int)

        self.triangles: List[Triangle] = []
        self.tri_indices: List[tuple] = []
        self.build_mesh()

        # compute per-vertex normals (smoothed)
        self.vertex_normals = np.zeros_like(self.vertices)
        self._compute_vertex_normals()

    def build_mesh(self):
        for (i1, i2, i3) in self.indices:
            tri = Triangle(self.vertices[i1], self.vertices[i2], self.vertices[i3])
            self.triangles.append(tri)
            self.tri_indices.append((i1, i2, i3))

    def _compute_vertex_normals(self):
        """
        Smooth normals = average of adjacent face normals.
        """
        self.vertex_normals[:] = 0.0
        for (i1, i2, i3), tri in zip(self.tri_indices, self.triangles):
            n = tri.normal
            self.vertex_normals[i1] += n
            self.vertex_normals[i2] += n
            self.vertex_normals[i3] += n

        # normalize
        for i in range(len(self.vertex_normals)):
            n = self.vertex_normals[i]
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                self.vertex_normals[i] = n / norm

    def __len__(self):
        return len(self.triangles)

    def __getitem__(self, idx):
        return self.triangles[idx]

    # ---------------------------------------------------------
    # NORMAL INTERPOLATION (using correct barycentric coords)
    # ---------------------------------------------------------
    def _interpolated_normal(self, tri_idx, bary):
        """
        bary = (u, v, w) from Triangle.closest_point
        """
        u, v, w = bary
        i1, i2, i3 = self.tri_indices[tri_idx]

        n = (
            u * self.vertex_normals[i1] +
            v * self.vertex_normals[i2] +
            w * self.vertex_normals[i3]
        )
        norm = np.linalg.norm(n)
        return n / norm

    # ---------------------------------------------------------
    # CLOSEST POINT (LINEAR SEARCH)
    # ---------------------------------------------------------
    def find_closest_point_linear(self, p):
        """
        Returns (closest_point, interpolated_normal)
        """
        p = np.asarray(p, float)
        min_dist = float('inf')
        best_cp = None
        best_bary = None
        best_idx = -1

        for idx, tri in enumerate(self.triangles):
            cp, bary = tri.closest_point(p)
            dist = np.linalg.norm(cp - p)
            if dist < min_dist:
                min_dist = dist
                best_cp = cp
                best_bary = bary
                best_idx = idx

        n = self._interpolated_normal(best_idx, best_bary)
        return best_cp, n

    # ---------------------------------------------------------
    # CLOSEST POINT (BOUNDING BOX ACCELERATION)
    # ---------------------------------------------------------
    def find_closest_point_box(self, p):
        """
        Returns (closest_point, interpolated_normal)
        """
        p = np.asarray(p, float)
        bound = float('inf')
        best_cp = None
        best_bary = None
        best_idx = -1

        # quick initial bound using first few triangles
        for idx, tri in enumerate(self.triangles[:6]):
            cp, bary = tri.closest_point(p)
            d = np.linalg.norm(cp - p)
            if d < bound:
                bound = d
                best_cp = cp
                best_bary = bary
                best_idx = idx

        # full pass with bounding-box pruning
        for idx, tri in enumerate(self.triangles):
            if not tri.in_box(p, bound):
                continue
            cp, bary = tri.closest_point(p)
            d = np.linalg.norm(cp - p)
            if d < bound:
                bound = d
                best_cp = cp
                best_bary = bary
                best_idx = idx

        n = self._interpolated_normal(best_idx, best_bary)
        return best_cp, n

    # ---------------------------------------------------------
    # VECTORIZED INTERFACE
    # ---------------------------------------------------------
    def find_closest_point(self, points, use_linear=False, return_normals=False):
        points = np.asarray(points, float)
        N = points.shape[0]
        out_points = np.zeros_like(points)

        if return_normals:
            out_normals = np.zeros_like(points)

        for i, p in enumerate(points):
            if use_linear:
                cp, n = self.find_closest_point_linear(p)
            else:
                cp, n = self.find_closest_point_box(p)

            out_points[i] = cp
            if return_normals:
                out_normals[i] = n

        if return_normals:
            return out_points, out_normals
        return out_points
