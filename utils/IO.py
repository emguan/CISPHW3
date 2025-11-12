import numpy as np

def read_xyz(line):
    comps = line.split()
    coord = []
    for i in comps: 
        coord.append(float(i))
    return coord

def read_body(filepath):

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    header = lines[0].split()
    N_markers = int(header[0])
    name = header[1]

    marker_lines = lines[1 : 1 + N_markers]
    markers = np.array([[float(x) for x in line.split()] for line in marker_lines])

    tip_line = lines[1 + N_markers]
    tip = np.array([float(x) for x in tip_line.split()])

    return markers, tip, N_markers, name

def read_mesh(filepath):
    
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    N_vertices = int(lines[0])

    vertex_lines = lines[1 : 1 + N_vertices]
    vertices = np.array([[float(x) for x in line.split()] for line in vertex_lines])

    N_triangles = int(lines[1 + N_vertices])

    tri_lines = lines[2 + N_vertices : 2 + N_vertices + N_triangles]
    data = np.array([[int(x) for x in line.split()] for line in tri_lines])

    triangle_indices = data[:, :3]
    neighbors = data[:, 3:]

    return vertices, N_vertices, N_triangles, triangle_indices, neighbors

def read_sample(filepath, N_A, N_B):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    parts = lines[0].split()
    N_s = int(parts[0])
    N_samps = int(parts[1])
    N_D = N_s - N_A - N_B

    A_samps = np.zeros((N_samps, N_A, 3))
    B_samps = np.zeros((N_samps, N_B, 3))

    idx = 1
    for s in range(N_samps):
        for i in range(N_A):
            A_samps[s, i] = [float(x) for x in lines[idx].split()]
            idx += 1
        for j in range(N_B):
            B_samps[s, j] = [float(x) for x in lines[idx].split()]
            idx += 1
        for _ in range(N_D):
            idx += 1

    return A_samps, B_samps, N_s, N_samps

    import numpy as np

def write_output(filename, D, C):
    N_samps = len(D)
    with open(filename, "w") as f:
        f.write(f"{N_samps} \"{filename}\"\n")
        for dk, ck in zip(D, C):
            diff = np.linalg.norm(dk - ck)
            f.write(f"{dk[0]:.6f} {dk[1]:.6f} {dk[2]:.6f} "
                    f"{ck[0]:.6f} {ck[1]:.6f} {ck[2]:.6f} "
                    f"{diff:.6f}\n")


if __name__ == "__main__":
    coords = read_body("./data/Problem3-BodyA.txt")
    print(coords)
    coords = read_mesh("./data/Problem3Mesh.sur")
    print(coords)
    #coords = read_sample("./data/PA3-A-Debug-SampleReadingsTest.sur")