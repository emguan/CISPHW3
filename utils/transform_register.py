"""
Transformation and registration utilities.
Author: Emily Guan (corrected & completed)
"""

import numpy as np


# ============================================================
# Arun's Method: Computes rigid transform A→B
# ============================================================
def aruns_method(A, B):
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Fix improper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


# ============================================================
# Compute pointer tip in frame B
# ============================================================
def compute_d(A_body_markers,
              B_body_markers,
              A_tip,
              A_samps,
              B_samps):

    N_samps = A_samps.shape[0]
    d = np.empty((N_samps, 3), float)

    for k in range(N_samps):
        Ra, ta = aruns_method(A_body_markers, A_samps[k])
        Rb, tb = aruns_method(B_body_markers, B_samps[k])

        tip_tracker = Ra @ A_tip + ta
        d[k] = Rb.T @ (tip_tracker - tb)

    return d


# ============================================================
# Utilities
# ============================================================
def apply(a, R, t):
    """Apply rigid transform."""
    return a @ R.T + t


def skew(p):
    """Return skew-symmetric matrix for cross-product."""
    return np.array([
        [0,     -p[2],  p[1]],
        [p[2],   0,    -p[0]],
        [-p[1],  p[0],  0]
    ])


def compute_Freg(mesh, d, threshold=0.05, max_iter=50):
    """
    Estimate F_reg such that c_k ≈ F_reg * d_k
    where d_k are sample points expressed in frame B.

    mesh must implement:
        mesh.find_closest_point(points) → (N x 3) closest points
    """

    # Initial guess (per Problem 3: identity, or passed-in initial)
    R = np.eye(3)
    t = np.zeros(3)

    for it in range(max_iter):

        # Step 1: apply transform to get sample points s_k
        s = apply(d, R, t)        # (N x 3)

        # Step 2: closest points on mesh
        c = mesh.find_closest_point(s)

        # Step 3: compute new transform mapping d → c
        R_delta, t_delta = aruns_method(d, c)

        # Step 4: update global transform
        R = R_delta @ R
        t = R_delta @ t + t_delta

        # Step 5: check convergence
        movement = np.linalg.norm(t_delta) + np.linalg.norm(R_delta - np.eye(3))
        if movement < threshold:
            break

    return R, t

def compute_ck(mesh, d, linear=False):
    """
    Problem 4:
        Estimate F_reg iteratively, then return
            c_k = closest mesh point to (F_reg * d_k)
    """

    R, t = compute_Freg(mesh, d)

    # Apply the final registration transform
    s = apply(d, R, t)

    # Return closest points on mesh
    return mesh.find_closest_point(s, use_linear=linear)
