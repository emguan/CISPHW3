"""
Tests for transform_register.py functions.
Author: Emily Guan 
"""

import numpy as np
from utils.transform_register import (
    apply,
    skew,
    aruns_method,
    compute_d,
    compute_Freg,
    compute_ck
)


# Utility helpers


def almost_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)


# Test skew()


def test_skew_matrix_properties():
    p = np.array([1.0, 2.0, 3.0])
    S = skew(p)

    # Should be antisymmetric: S^T = –S
    assert almost_equal(S.T, -S), "Skew matrix is not antisymmetric"

    # Check that S @ p == 0
    assert almost_equal(S @ p, np.zeros(3)), "Skew(p) * p must be zero"


# Test apply()


def test_apply_transform():
    R = np.eye(3)
    t = np.array([5, -2, 10])
    p = np.array([[1,2,3], [0,0,0]])

    q = apply(p, R, t)

    expected = p + t
    assert almost_equal(q, expected), "apply() failed with identity rotation"


# Test compute_d()
def test_compute_d_simple():
    """
    Construct a situation where:
        F_Ak = identity
        F_Bk = identity
    -> d_k = tipA
    """
    A_markers = np.array([[0,0,0], [1,0,0], [0,1,0]])
    B_markers = A_markers.copy()
    A_tip = np.array([2,3,4])

    # samples identical to markers
    A_samps = np.tile(A_markers, (5,1,1))
    B_samps = A_samps.copy()

    d = compute_d(A_markers, B_markers, A_tip, A_samps, B_samps)

    for row in d:
        assert almost_equal(row, A_tip), "compute_d incorrect when transforms are identity"


# Mock mesh for testing compute_Freg()
class MockMesh:
    def find_closest_point(self, p, use_linear=False, return_normals=False):
        p = np.asarray(p)
        if return_normals:
            normals = np.tile(np.array([0,0,1.0]), (p.shape[0],1))
            return p.copy(), normals
        return p.copy()

# Test compute_Freg()
def test_compute_Freg_identity():
    mesh = MockMesh()

    d = np.array([
        [0,0,0],
        [1,2,3],
        [-5,1,4]
    ])

    R, t = compute_Freg(mesh, d, threshold=1e-6, max_iter=5)

    assert almost_equal(R, np.eye(3)), "compute_Freg should yield identity R"
    assert almost_equal(t, np.zeros(3)), "compute_Freg should yield zero t"


# Test compute_ck()
def test_compute_ck_identity_reg():
    mesh = MockMesh()

    d = np.array([
        [1,2,3],
        [4,5,6]
    ])

    (c, s) = compute_ck(mesh, d, max_iter=50, threshold=1e-2)

    # s = R*d + t = d
    assert almost_equal(s, d), "s_k should equal d_k when registration=identity"

    # c = closest point = s = d
    assert almost_equal(c, d), "c_k should equal d_k when mesh reflects identity projection"


# Test runner
def main():
    tests = [
        test_skew_matrix_properties,
        test_apply_transform,
        test_compute_d_simple,
        test_compute_Freg_identity,
        test_compute_ck_identity_reg,
    ]

    print("\nRunning transform_register tests...\n")

    passed = failed = 0

    for test in tests:
        try:
            test()
            print(f"Passed {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"Failed {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"Failed {test.__name__}: unexpected error → {e}")
            failed += 1

    print(f"\nPassed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\nAll transform_register tests passed!")

if __name__ == "__main__":
    main()
