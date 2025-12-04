"""
Reads and writes all files according to PAHW3 descriptions. 

Author: Emily Guan
"""

import numpy as np



"""
Reads Problem3-Body text file.

INput:
    filepath: str
        Path to a body definition file.

File:
    Line 1: <N_markers> <body_name>
    Next N: LED marker coordinates (x y z)
    Last line: tip coordinates (x y z)

Returns:
    markers: (N_markers x 3) numpy array
        Coordinates of body markers in body coordinate frame.
    tip: (3,) numpy array
        Tip offset coordinates in body frame.
    N_markers: int
        Number of markers found.
    name: str
        Name of the body.

"""
def read_body(filepath):

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    header = lines[0].split()
    N_markers = int(header[0])
    name = header[1]

    marker_lines = lines[1: 1 + N_markers]
    markers = np.array([[float(x) for x in line.split()] for line in marker_lines])

    tip_line = lines[1 + N_markers]
    tip = np.array([float(x) for x in tip_line.split()])

    return markers, tip, N_markers, name

"""
Reads surface mesh file (.sur).

Input:
    filepath: str
        Path to the mesh file.

File:
    Line 1: <N_vertices>
    Next N_vertices: vertex coordinates (x y z)
    Next line:  <N_triangles>
    Next N_triangles: <v1> <v2> <v3> <neighbor1> <neighbor2> <neighbor3>

Return:
    vertices: (N_vertices x 3) numpy array
    N_vertices: int
    N_triangles: int
    triangle_indices: (N_triangles x 3) numpy array of ints
        Indices of each triangle’s 3 vertices.
    neighbors: (N_triangles x 3) numpy array of ints
        Neighbor indices for each triangle (unused here).
"""
def read_mesh(filepath):
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    N_vertices = int(lines[0])

    vertex_lines = lines[1: 1 + N_vertices]
    vertices = np.array([[float(x) for x in line.split()] for line in vertex_lines])

    N_triangles = int(lines[1 + N_vertices])

    tri_lines = lines[2 + N_vertices: 2 + N_vertices + N_triangles]
    data = np.array([[int(x) for x in line.split()] for line in tri_lines])

    triangle_indices = data[:, :3]
    neighbors = data[:, 3:]

    return vertices, N_vertices, N_triangles, triangle_indices, neighbors

"""
Reads sample (tracker) data.

Input:
    filepath: str
        Path to sample readings file.
    N_A: int
        Number of markers on body A.
    N_B: int
        Number of markers on body B.

File:
    Line 1: <N_total_markers>,<N_samples>
    For each sample k:
        N_A lines: coordinates of A markers 
        N_B lines: coordinates of B markers
        N_D lines: coordinates of D markers

Output:
    A_samps: (N_samples x N_A x 3) array
        Tracker coordinates of body A markers over time.
    B_samps: (N_samples x N_B x 3) array
        Tracker coordinates of body B markers over time.
    N_s: int
        Total number of markers (A + B + D).
    N_samps: int
        Number of sample frames.
"""
def read_sample(filepath, N_A, N_B):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    parts = lines[0].split(',')
    N_s = int(parts[0])
    N_samps = int(parts[1])
    N_D = N_s - N_A - N_B

    A_samps = np.zeros((N_samps, N_A, 3))
    B_samps = np.zeros((N_samps, N_B, 3))

    idx = 1
    for s in range(N_samps):
        for i in range(N_A):
            A_samps[s, i] = [float(x) for x in lines[idx].split(',')]
            idx += 1
        for j in range(N_B):
            B_samps[s, j] = [float(x) for x in lines[idx].split(',')]
            idx += 1
        for _ in range(N_D):
            idx += 1

    return A_samps, B_samps, N_s, N_samps

"""
Writes PAHW3 output file.

Input:
    filename: str
        Path where the output should be written.
    S: (N_samples x 3) array
        s_k values: estimated tip positions in B-frame.
    C: (N_samples x 3) array
        c_k values: corresponding points on the mesh.

Returns:
    Writes a formatted text file containing:
        Line 1:  <N_samples> <filename>
        Then for each sample k: dk_x dk_y dk_z  ck_x ck_y ck_z  |dk - ck|
"""
def write_output(filename, S, C):
    N_samps = len(S)
    with open(filename, "w") as f:
        f.write(f"{N_samps} {filename}\n")
        for sk, ck in zip(S, C):
            diff = np.linalg.norm(sk - ck)

            # Force tiny values to 0.000
            if diff < 0.0005:
                diff = 0.0

            f.write(f"{sk[0]:9.2f} {sk[1]:9.2f} {sk[2]:9.2f} "
                    f"{ck[0]:9.2f} {ck[1]:9.2f} {ck[2]:9.2f} "
                    f"{diff:9.3f}\n")