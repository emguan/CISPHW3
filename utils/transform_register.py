"""
Transformation and registration utilities.
Author: Emily Guan (corrected & completed)
"""

import numpy as np

"""
Compute the rigid transformation (R, p) that aligns point set A to B
such that:  b_i ≈ R * a_i + p

Using Arun's method: https://jingnanshi.com/blog/arun_method_for_3d_reg.html 
"""
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


"""
Given a vector, R, t, applies a rigid transform.
"""
def apply(a, R, t):
    """Apply rigid transform."""
    return a @ R.T + t

"""
Return skew-symmetric matrix for cross-product.
"""
def skew(p):
    return np.array([
        [0,     -p[2],  p[1]],
        [p[2],   0,    -p[0]],
        [-p[1],  p[0],  0]
    ])

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
def compute_Freg(mesh, d, threshold=1e-3, max_iter=100, rho=0.05, use_linear=False):


    # Initial guess
    R = np.eye(3)
    t = np.zeros(3)

    N = d.shape[0]

    for it in range(max_iter):
        print("iteration ", it)
        # p_i~ = R d_i + t
        p = apply(d, R, t)  # (N,3)

        c, normals = mesh.find_closest_point(p,
                                             return_normals=True)

        # Build linearized least squares A x ≈ b
        # x = [u_tilde (3,); delta_t (3,)]
        A = np.zeros((3 * N, 6))
        b = np.zeros(3 * N)

        row = 0
        for i in range(N):
            pi = p[i]
            ci = c[i]
            vi = normals[i]

            nrm = np.linalg.norm(vi)
            if nrm > 0:
                vi = vi / nrm

            Pi = skew(pi)
            Vi = skew(vi)

            Ai = np.hstack((2.0 * Vi @ Pi, Vi))
            bi = Vi @ (ci - pi)

            A[row:row+3, :] = Ai
            b[row:row+3] = bi
            row += 3

        # Solve least squares
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        u_tilde = x[0:3]
        delta_t = x[3:6]

        # ||u_tilde|| <= rho
        norm_u = np.linalg.norm(u_tilde)
        if norm_u > rho:
            u_tilde = (rho / norm_u) * u_tilde

        # ΔR = (I - U)(I + U)^{-1}, U = skew(u_tilde)
        U = skew(u_tilde)
        I = np.eye(3)
        DeltaR = (I - U) @ np.linalg.inv(I + U)

        # Update
        R = DeltaR @ R
        t = DeltaR @ t + delta_t

        # epsilon = average residual in LS system
        x_full = np.concatenate([u_tilde, delta_t])
        residual = A @ x_full - b
        eps = np.linalg.norm(residual) / np.sqrt(N)
        print(eps)

        if eps < threshold:
            break

    return R, t


def compute_ck(mesh, d, linear=False):

    R, t = compute_Freg(mesh, d)

    s = apply(d, R, t)

    return mesh.find_closest_point(s), s
