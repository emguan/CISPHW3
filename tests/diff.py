import numpy as np

def compute_errors(our_file, ref_file):
    data_ours = np.loadtxt(our_file, skiprows=1)
    data_ref = np.loadtxt(ref_file, skiprows=1)

    c_ours = data_ours[:, 3:6]  # Extract c_k columns
    c_ref = data_ref[:, 3:6]

    errors = np.linalg.norm(c_ours - c_ref, axis=1)
    return np.mean(errors), np.max(errors)

for dataset in ['A', 'B', 'C', 'D', 'E', 'F']:
    mean, max_err = compute_errors(
        f'output/pa3-{dataset}-output.txt',
        f'data/PA3-{dataset}-Debug-Output.txt')
    print(f"{dataset} & {mean:.4f} & {max_err:.4f}")