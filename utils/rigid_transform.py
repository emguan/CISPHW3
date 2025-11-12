import numpy as np
"""
Compute the rigid transformation (R, p) that aligns point set A to B
such that:  b_i ≈ R * a_i + p

Using Arun's method: https://jingnanshi.com/blog/arun_method_for_3d_reg.html 
"""
def aruns_method(A: np.ndarray, B: np.ndarray):

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

def compute_d(A_body_markers: np.ndarray,
                     B_body_markers: np.ndarray,
                     A_tip: np.ndarray,
                     A_samps: np.ndarray,
                     B_samps: np.ndarray):
    
    N_samps = A_samps.shape[0]
    D = np.empty((N_samps, 3), dtype=float)

    for k in range(N_samps):
        Ra, ta = aruns_method(A_body_markers, A_samps[k]) 
        Rb, tb = aruns_method(B_body_markers, B_samps[k]) 

        tip_in_tracker = Ra @ A_tip + ta
        d_k = Rb.T @ (tip_in_tracker - tb)
        D[k] = d_k
    return D

import numpy as np

def apply_Freg(points: np.ndarray, R=None, t=None) -> np.ndarray:

    if R is None and t is None:
        return points
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    return (points @ R.T) + t  # (N,3)

def compute_ck(mesh, d_series: np.ndarray, R=None, t=None) -> np.ndarray:
    """
    For each d_k, compute s_k = F_reg * d_k and then c_k = closest point on mesh to s_k.
    mesh must expose .find_closest_point(p) or .closest_point(p) -> (3,)
    Returns C: (N_samps, 3)
    """
    s_series = apply_Freg(d_series, R, t)
    C = np.empty_like(s_series, dtype=float)
    use_find = hasattr(mesh, "find_closest_point")
    for k, s_k in enumerate(s_series):
        p = mesh.find_closest_point(s_k) if use_find else mesh.closest_point(s_k)
        C[k] = p
    return C
