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


def compute_Freg(mesh, d, threshold=1e-3, max_iter=100, rho=0.05, use_linear=False):
    """
    Constrained linearized least-squares registration (Gueziec et al. 1998).

    Estimate F_reg such that c_k ≈ F_reg * d_k, where d_k are sample
    points expressed in frame B (shape N x 3).

    mesh must implement:
        mesh.find_closest_point(points, use_linear=False, return_normals=True)
            → (closest_points (N x 3), normals (N x 3))

    Parameters
    ----------
    mesh : Mesh
    d : (N,3) array
        Sample points in frame B.
    threshold : float
        Termination threshold on residual (epsilon).
    max_iter : int
        Maximum number of Gauss–Newton iterations.
    rho : float
        Upper bound on ||u_tilde|| (small rotation update, in radians).
    use_linear : bool
        If True, use linear closest-point search instead of box search.

    Returns
    -------
    R : (3,3) array
    t : (3,)   array
    """

    # Initial guess (Problem 3: identity)
    R = np.eye(3)
    t = np.zeros(3)

    N = d.shape[0]

    for it in range(max_iter):
        print("iteration ", it)
        # Step 1: transformed sample points p_i~ = R d_i + t
        p = apply(d, R, t)  # (N,3)

        # Closest points on mesh and associated normals v_i
        c, normals = mesh.find_closest_point(p,
                                             return_normals=True)

        # Build linearized least squares system A x ≈ b
        # where x = [u_tilde (3,); delta_t (3,)]
        A = np.zeros((3 * N, 6))
        b = np.zeros(3 * N)

        row = 0
        for i in range(N):
            pi = p[i]
            ci = c[i]
            vi = normals[i]

            # ensure vi is unit length (defensive)
            nrm = np.linalg.norm(vi)
            if nrm > 0:
                vi = vi / nrm

            Pi = skew(pi)        # 3x3
            Vi = skew(vi)        # 3x3

            Ai = np.hstack((2.0 * Vi @ Pi, Vi))  # 3x6
            bi = Vi @ (ci - pi)                  # 3,

            A[row:row+3, :] = Ai
            b[row:row+3] = bi
            row += 3

        # Solve least squares: A x ≈ b
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        u_tilde = x[0:3]
        delta_t = x[3:6]

        # Enforce ||u_tilde|| <= rho
        norm_u = np.linalg.norm(u_tilde)
        if norm_u > rho:
            u_tilde = (rho / norm_u) * u_tilde

        # Cayley map: ΔR = (I - U)(I + U)^{-1}, U = skew(u_tilde)
        U = skew(u_tilde)
        I = np.eye(3)
        DeltaR = (I - U) @ np.linalg.inv(I + U)

        # Update global transform
        R = DeltaR @ R
        t = DeltaR @ t + delta_t


        # Compute epsilon = average residual in LS system
        x_full = np.concatenate([u_tilde, delta_t])
        residual = A @ x_full - b
        eps = np.linalg.norm(residual) / np.sqrt(N)
        print(eps)

        # Termination
        if eps < threshold:
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
    return mesh.find_closest_point(s), s
