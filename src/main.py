import argparse

from utils.IO import read_body, read_mesh, read_sample, write_output
from utils.mesh import Mesh
from utils.rigid_transform import compute_d, compute_ck

def main(A_file, B_file, mesh_file, sample_file, outfile): 

    markersA, tipA, NA, nameA = read_body(A_file)
    markersB, tipB, NB, nameB = read_body(B_file)
    vertices, N_vertices, N_triangles, triangle_indices, neighbors = read_mesh(mesh_file)
    A_samps, B_samps, N_s, N_samps = read_sample(sample_file, NA, NB)

    mesh = Mesh(vertices, triangle_indices)

    d = compute_d(markersA, markersB, tipA, A_samps, B_samps)

    c = compute_ck(mesh, d)

    write_output(outfile, d, c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA3")
    parser.add_argument("--A", required=True)
    parser.add_argument("--B", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    main(args.A, args.B, args.mesh, args.sample, args.out)

'''
python src/main.py --A data/Problem3-BodyA.txt --B data/Problem3-BodyB.txt --mesh data/Problem3Mesh.sur --sample data/PA3-A-Debug-SampleReadingsTest.txt --out output/pa3-A-output.txt
'''