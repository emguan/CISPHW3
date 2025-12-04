"""
Computing errors between our file and debug files.

Author: Brian Sun
"""

import numpy as np

"""Computes difference between our files and debug."""
def compute_errors(our_file, ref_file):
    data_ours = np.loadtxt(our_file, skiprows=1)
    data_ref = np.loadtxt(ref_file, skiprows=1)
    
    s_ours = data_ours[:, 0:3]  # Extract s_kc_k columns
    s_ref = data_ref[:, 0:3]

    c_ours = data_ours[:, 3:6]  # Extract c_k columns
    c_ref = data_ref[:, 3:6]

    e_ours = data_ours[:, 6]  # Extract error columns
    e_ref = data_ref[:, 6]

    serrors = np.linalg.norm(s_ours - s_ref, axis=1)
    cerrors = np.linalg.norm(c_ours - c_ref, axis=1)
    eerrors = np.linalg.norm(e_ours - e_ref)
    return np.mean(serrors), np.max(serrors), np.mean(cerrors), np.max(cerrors), np.mean(eerrors), np.max(eerrors)

for dataset in ['A', 'B', 'C', 'D', 'E', 'F']:
    cmean, cmax, smean, smax, emean, emax= compute_errors(
        f'output/PA4-{dataset}-output.txt',
        f'data/PA4-{dataset}-Debug-Output.txt')
    print(f"{dataset} & {cmean:.4f} & {cmax:.4f}& {smean:.4f} & {smax:.4f}& {emean:.4f} & {emax:.4f}")
    print(f"{dataset} & {cmean:.4f} & {cmax:.4f}& {smean:.4f} & {smax:.4f}& {emean:.4f} & {emax:.4f}")