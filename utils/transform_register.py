"""
Contains transformation and registration functions. 

Author: Emily Guan
"""

import numpy as np

"""
Compute the rigid transformation (R, p) that aligns point set A to B
such that:  b_i ≈ R * a_i + p.

Taken from PAHW1. 

Using Arun's method: https://jingnanshi.com/blog/arun_method_for_3d_reg.html 

Input:
    A: (N x 3) numpy array
        Source point set.

    B: (N x 3) numpy array
        Target point set.

Output:
    R: (3 x 3) numpy array
        Rotation matrix.

    t: (3,) numpy array
        Translation vector.
"""
def aruns_method(A, B):

    # find centroids
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)

    # center the point clouds
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # compute H
    H = A_centered.T @ B_centered

    # orthogonalize to get R0 & "iterate" via svd
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # handle reflection (verify is rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # compute translation
    t = centroid_B - R @ centroid_A

    return R, t

"""
Computes position d of pointer tip w.r.t. B. 

Follows eqn: d = F_Bk^-1 * F_Ak * A_tip

Input:
    A_body_markers: (N_A x 3)
        Body A marker coordinates in A's frame.

    B_body_markers: (N_B x 3)
        Body B marker coordinates in B's frame.

    A_tip: (3,)
        Tip offset in body A coordinates.

    A_samps: (N_samples x N_A x 3)
        Tracker coordinates of A markers for each sample.

    B_samps: (N_samples x N_B x 3)
        Tracker coordinates of B markers for each sample.

Output:
    d: (N_samples x 3)
        Pointer tip position expressed in B's coordinate frame.
"""
def compute_d(A_body_markers,
                     B_body_markers,
                     A_tip,
                     A_samps,
                     B_samps):
    
    N_samps = A_samps.shape[0]
    d = np.empty((N_samps, 3), dtype=float)

    for k in range(N_samps):

        Ra, ta = aruns_method(A_body_markers, A_samps[k]) # F_Ak
        Rb, tb = aruns_method(B_body_markers, B_samps[k]) # F_Bk


        tip_in_tracker = Ra @ A_tip + ta # F_Ak * A_tip
        d_k = Rb.T @ (tip_in_tracker - tb) # F_Bk^-1 * F_Ak * A_tip
        d[k] = d_k

    return d

"""
Given transformation [R,t], apply it to point. 

Made this for safe calculations.

Input:
    points: (N x 3) or (3,) array
        Points to transform.

    R: (3 x 3) matrix or None
        Rotation. If None, identity is used.

    t: (3,) vector or None
        Translation. If None, zero is used.

Output:
    transformed_points: array of same shape as input
"""
def apply_Freg(points, R=None, t=None):
    if R is None and t is None:
        return points
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    return (points @ R.T) + t

"""
Computes c. 

Finds closest points on mesh to F_reg * d. 

Input:
    mesh: Mesh
        Mesh object containing triangle surfaces.

    d_series: (N_samples x 3)
        d_k values (pointer tip expressed in B).

    R, t: optional transformation
        Allows applying an additional rigid registration to d_k.

    linear: bool
        If True, uses full linear search (slow but accurate).
        If False, uses bounding-box optimized search.

Output:
    C: (N_samples x 3)
        Closest mesh points for each sample.
"""
def compute_ck(mesh, d_series, R=None, t=None, linear = False):
    s_series = apply_Freg(d_series, R, t) # F_reg * d
    C = np.empty_like(s_series, dtype=float)
    for k, s_k in enumerate(s_series):
        if linear: 
            p = mesh.find_closest_point_linear(s_k)
        else: 
            p = mesh.find_closest_point(s_k)
        C[k] = p
    return C
