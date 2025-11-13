"""
Tests all functions under Triangle class.

Author: Emily Guan
"""

import numpy as np
from utils.triangles import Triangle 

"""
Accuracy check function. Gives tolerance of 10^-6.
"""
def almost_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)

"""
Checks correct normal is unit length.
"""
def test_normal_unit_length():
    tri = Triangle([0, 0, 0], [1, 0, 0], [0, 1, 0])
    assert almost_equal(np.linalg.norm(tri.normal), 1.0), "normal is not unit length"

"""
Checks signs of normal is correct.
"""
def test_normal_direction():
    tri = Triangle([0, 0, 0], [1, 0, 0], [0, 1, 0])
    assert almost_equal(tri.normal, [0, 0, 1]), "normal direction incorrect"

"""
Checks upper & lower bound calculations.
"""
def test_bounds():
    tri = Triangle([1, 2, 3], [4, -1, 10], [0, 5, 7])
    assert almost_equal(tri.lb, [0, -1, 3]), "lower bounds incorrect"
    assert almost_equal(tri.ub, [4, 5, 10]), "upper bounds incorrect"

"""
Checks barycentric coords are calculated correctly (checks sum to 1).
"""
def test_barycentric_center():
    tri = Triangle([0, 0, 0], [2, 0, 0], [0, 2, 0])
    p = np.array([2/3, 2/3, 0])
    u, v, w = tri.barycentric_coords(p)

    assert almost_equal(u + v + w, 1.0), "barycentric sum != 1"
    assert u >= 0 and v >= 0 and w >= 0, "barycentric coords not all positive"

"""
Checks projection function.
"""
def test_project_to_plane():
    tri = Triangle([0, 0, 0], [1, 0, 0], [0, 1, 0])
    p = np.array([0.3, 0.3, 5.0])
    proj = tri.project_to_plane(p)
    assert almost_equal(proj, [0.3, 0.3, 0.0]), "projection to plane incorrect"

"""
Checks in_box function will correctly detect point in box given margin.

Expects true.
"""
def test_in_box_true():
    tri = Triangle([0, 0, 0], [1, 2, 3], [-1, -2, -3])
    p = np.array([0.5, 0.5, 0.5])
    assert tri.in_box(p, 0.1), "point should be inside box"

"""
Checks in_box function will correctly detect point in box given margin.

Expects false.
"""
def test_in_box_false():
    tri = Triangle([0, 0, 0], [1, 2, 3], [-1, -2, -3])
    p = np.array([10, 10, 10])
    assert not tri.in_box(p, 0.1), "Point should be outside box."

"""
Checks case where closest projected point is within triangle, ensures no change.
"""
def test_closest_point_inside_triangle():
    tri = Triangle([0, 0, 0], [1, 0, 0], [0, 1, 0])
    p = np.array([0.2, 0.2, 1.0])
    c = tri.closest_point(p)
    assert almost_equal(c, [0.2, 0.2, 0]), "Closest point (inside) incorrect."

"""
Checks case where closest projected point is not within triangle and not on vertex.
"""
def test_closest_point_near_edge():
    tri = Triangle([0, 0, 0], [2, 0, 0], [0, 2, 0])
    p = np.array([1.0, -1.0, 0])
    c = tri.closest_point(p)
    assert almost_equal(c, [1.0, 0.0, 0.0]), "Closest point (edge) incorrect."

"""
Checks case where closest projected point is on vertex. 
"""
def test_closest_point_near_vertex():
    tri = Triangle([0, 0, 0], [2, 0, 0], [0, 2, 0])
    p = np.array([-1.0, -1.0, 0])
    c = tri.closest_point(p)
    assert almost_equal(c, [0.0, 0.0, 0.0]), "Closest point (vertex) incorrect."


def main():
    tests = [
        test_normal_unit_length,
        test_normal_direction,
        test_bounds,
        test_barycentric_center,
        test_project_to_plane,
        test_in_box_true,
        test_in_box_false,
        test_closest_point_inside_triangle,
        test_closest_point_near_edge,
        test_closest_point_near_vertex,
    ]

    print("Running Triangle class tests...\n")

    passed = 0
    failed = 0

    for t in tests:
        try:
            t()
            print(f"✔ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✘ {t.__name__}: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"✘ {t.__name__}: unexpected error: {e}")
            failed += 1

    print(f" Passed: {passed}")
    print(f" Failed: {failed}")

    if failed == 0:
        print("\nAll Triangle tests passed")


if __name__ == "__main__":
    main()
