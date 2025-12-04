"""
Tests all functions under Mesh class.

Author: Emily Guan
"""

import numpy as np
from utils.mesh import Mesh

def almost_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)

def test_mesh_build():
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    indices = [(0,1,2),(0,1,3)]
    mesh = Mesh(vertices, indices)
    assert len(mesh) == 2, "mesh did not build correct number of triangles"

def test_getitem():
    vertices = [[0,0,0],[1,0,0],[0,1,0]]
    mesh = Mesh(vertices, [(0,1,2)])
    tri = mesh[0]
    assert almost_equal(tri.a, [0,0,0]), "__getitem__ returned wrong triangle."

def test_find_closest_point_linear_inside_triangle():
    vertices = [[0,0,0],[1,0,0],[0,1,0]]
    mesh = Mesh(vertices, [(0,1,2)])
    p = [0.2, 0.2, 5]

    cp, n = mesh.find_closest_point_linear(p)
    assert almost_equal(cp, [0.2,0.2,0]), "incorrect closest point for inside-triangle case"

def test_find_closest_point_linear_outside_edge():
    vertices = [[0,0,0],[2,0,0],[0,2,0]]
    mesh = Mesh(vertices, [(0,1,2)])
    p = [1, -1, 0]

    cp, n = mesh.find_closest_point_linear(p)
    assert almost_equal(cp, [1,0,0]), "closest point to edge incorrect"

def test_find_closest_point_linear_outside_vertex():
    vertices = [[0,0,0],[2,0,0],[0,2,0]]
    mesh = Mesh(vertices, [(0,1,2)])
    p = [-1, -1, 0]

    cp, n = mesh.find_closest_point_linear(p)
    assert almost_equal(cp, [0,0,0]), "closest point to vertex incorrect"

def test_find_closest_point_multiple_triangles():
    vertices = [
        [0,0,0], [1,0,0], [0,1,0],
        [5,5,0], [6,5,0], [5,6,0],
    ]
    indices = [(0,1,2),(3,4,5)]
    mesh = Mesh(vertices, indices)

    p = [0.2, 0.2, 1.0]

    # linear version returns (closest_pt, normal)
    cp1, _ = mesh.find_closest_point_linear(p)

    # bounded version returns (N,3) point array
    cp2 = mesh.find_closest_point([p], return_normals=False)[0]

    assert almost_equal(cp1, [0.2,0.2,0]), "linear search wrong"
    assert almost_equal(cp2, [0.2,0.2,0]), "bounded search wrong"

def test_find_closest_point_matches_linear():
    vertices = [
        [0,0,0],[1,0,0],[0,1,0],
        [2,0,0],[3,0,0],[2,1,0],
        [0,0,2],[1,0,2],[0,1,2],
    ]
    indices = [(0,1,2),(3,4,5),(6,7,8)]
    mesh = Mesh(vertices, indices)

    test_points = [
        [0.2, 0.2, 5],
        [2.5, -1, 0],
        [1, 1, 3],
        [-1, 0.3, 0],
        [3, 0.2, 0.5]
    ]

    for p in test_points:
        cp_lin, _ = mesh.find_closest_point_linear(p)

        cp_box = mesh.find_closest_point([p], return_normals=False)[0]

        assert almost_equal(cp_lin, cp_box), f"mismatch for point {p}"

def main():
    tests = [
        test_mesh_build,
        test_getitem,
        test_find_closest_point_linear_inside_triangle,
        test_find_closest_point_linear_outside_edge,
        test_find_closest_point_linear_outside_vertex,
        test_find_closest_point_multiple_triangles,
        test_find_closest_point_matches_linear,
    ]

    print("\nRunning Mesh tests...\n")
    passed = failed = 0

    for t in tests:
        try:
            t()
            print(f"✔ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✘ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✘ {t.__name__}: unexpected error → {e}")
            failed += 1

    print(f"\nPassed: {passed}")
    print(f"Failed: {failed}")
    if failed == 0:
        print("All Mesh tests passed!")

if __name__ == "__main__":
    main()
