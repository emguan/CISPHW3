import argparse

from utils.IO import read_body, read_mesh
from utils.mesh import Mesh

def main(A_file, B_file, mesh_file): 

    A_coords, NA = read_body(A_file) 
    B_coords, NB = read_body(B_file)
    vertices, N_verts, N_triangles, vert_idxs = read_mesh(mesh_file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA3")
    parser.add_argument("--A", required=True)
    parser.add_argument("--B", required=True)
    parser.add_argument("--mesh", required=True)
    args = parser.parse_args()

    main(args.A, args.B, args.mesh)
