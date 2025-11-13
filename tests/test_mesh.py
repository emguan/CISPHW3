"""
Tests all functions under Mesh class.

Author: Emily Guan
"""

import numpy as np
from utils.mesh import Mesh

"""
Accuracy check function. Gives tolerance of 10^-6.
"""
def almost_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)

"""
Checks correctness of mesh build (num triangles).
"""
def test_mesh_build():
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    indices = [
        (0, 1, 2),
        (0, 1, 3),
    ]
    mesh = Mesh(vertices, indices)

    assert len(mesh) == 2, "mesh did not build correct number of triangles"

"""
Checks correctness of retrieval function. 
"""
def test_getitem():
    vertices = [[0,0,0],[1,0,0],[0,1,0]]
    mesh = Mesh(vertices, [(0,1,2)])
    tri = mesh[0]
    assert almost_equal(tri.a, [0,0,0]), "__getitem__ returned wrong triangle."

"""
Checks case where closest projected point is within triangle using simple model.
"""
def test_find_closest_point_linear_inside_triangle():
    vertices = [[0,0,0],[1,0,0],[0,1,0]]
    mesh = Mesh(vertices, [(0,1,2)])
    p = [0.2, 0.2, 5]
    cp = mesh.find_closest_point_linear(p)
    assert almost_equal(cp, [0.2,0.2,0]), "incorrect closest point for inside-triangle case"

"""
Checks case where closest projected point is on edge using simple model.
"""
def test_find_closest_point_linear_outside_edge():
    vertices = [[0,0,0],[2,0,0],[0,2,0]]
    mesh = Mesh(vertices, [(0,1,2)])
    p = [1, -1, 0]
    cp = mesh.find_closest_point_linear(p)
    assert almost_equal(cp, [1,0,0]), "closest point to edge incorrect"

"""
Checks case where closest projected point is on vertex using simple model.
"""
def test_find_closest_point_linear_outside_vertex():
    vertices = [[0,0,0],[2,0,0],[0,2,0]]
    mesh = Mesh(vertices, [(0,1,2)])
    p = [-1, -1, 0]
    cp = mesh.find_closest_point_linear(p)
    assert almost_equal(cp, [0,0,0]), "closest point to vertex incorrect"

"""
Checks method will choose closest triangle amongst list of close triangles.
"""
def test_find_closest_point_multiple_triangles():
    vertices = [ # two triangles on opposite sides
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],

        [5, 5, 0],
        [6, 5, 0],
        [5, 6, 0],
    ]
    indices = [(0,1,2), (3,4,5)]
    mesh = Mesh(vertices, indices)

    p = [0.2, 0.2, 1.0]  # close to first triangle
    cp = mesh.find_closest_point_linear(p)
    assert almost_equal(cp, [0.2, 0.2, 0]), "linear search failed on multi-triangle mesh"

    cp2 = mesh.find_closest_point(p)
    assert almost_equal(cp2, [0.2, 0.2, 0]), "bounded search failed on multi-triangle mesh"

"""
Checks linear and bounded box both agree, not checking for accuracy.. 
"""
def test_find_closest_point_matches_linear():
    vertices = [
        [0,0,0], [1,0,0], [0,1,0],
        [2,0,0], [3,0,0], [2,1,0],
        [0,0,2], [1,0,2], [0,1,2],
    ]
    indices = [(0,1,2), (3,4,5), (6,7,8)]
    mesh = Mesh(vertices, indices)

    test_points = [
        [0.2, 0.2, 5],
        [2.5, -1, 0],
        [1, 1, 3],
        [-1, 0.3, 0],
        [3, 0.2, 0.5]
    ]

    for p in test_points:
        linear = mesh.find_closest_point_linear(p)
        bounded = mesh.find_closest_point(p)

        assert almost_equal(linear, bounded), f"mismatch between linear and bounded search for point {p}"


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

    passed = 0
    failed = 0

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

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    if failed == 0:
        print("\nAll Mesh tests passed")


if __name__ == "__main__":
    main()
