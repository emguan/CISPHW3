from typing import List, Tuple
import numpy as np
from utils.triangles import Triangle

class Mesh:
    def __init__(self, vertices, indices):
        self.triangles: List[Triangle] = []
        self.build_mesh(vertices, indices)

    def build_mesh(self, vertices, indices):
        vertices = [np.array(v, dtype=float) for v in vertices]

        for (i1, i2, i3) in indices:
            tri = Triangle(vertices[i1], vertices[i2], vertices[i3])
            self.triangles.append(tri)

    def __len__(self):
        return len(self.triangles)

    def __getitem__(self, idx):
        return self.triangles[idx]

    def find_closest_point(self, p):
        p = np.array(p, dtype=float)
        closest_point = None
        min_dist = float('inf')

        for tri in self.triangles:
            candidate = tri.closest_point(p)
            dist = np.linalg.norm(candidate - p)
            if dist < min_dist:
                min_dist = dist
                closest_point = candidate
        
        return closest_point
