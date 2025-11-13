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
