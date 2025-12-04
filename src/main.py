"""
Execution source file for PA3. 

Examples Usage: 
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-A-Debug-SampleReadingsTest.txt --out output/pa4-A-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-B-Debug-SampleReadingsTest.txt --out output/pa4-B-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-C-Debug-SampleReadingsTest.txt --out output/pa4-C-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-D-Debug-SampleReadingsTest.txt --out output/pa4-D-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-E-Debug-SampleReadingsTest.txt --out output/pa4-E-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-F-Debug-SampleReadingsTest.txt --out output/pa4-F-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-G-Unknown-SampleReadingsTest.txt --out output/pa4-G-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-H-Unknown-SampleReadingsTest.txt --out output/pa4-H-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-J-Unknown-SampleReadingsTest.txt --out output/pa4-J-output.txt
python src/main.py --A data/Problem4-BodyA.txt --B data/Problem4-BodyB.txt --mesh data/Problem4MeshFile.sur --sample data/PA4-K-Unknown-SampleReadingsTest.txt --out output/pa4-K-output.txt


Author: Emily Guan
"""

import argparse

from utils.IO import read_body, read_mesh, read_sample, write_output
from utils.mesh import Mesh
from utils.transform_register import compute_d, compute_ck

"""
Full workflow run.

Inputs:
    A_file      - Body A definition file.
    B_file      - Body B definition file.
    mesh_file   - Surface mesh file.
    sample_file - Sampled marker readings for body A & B over multiple frames.
    outfile     - Output filepath for writing d_k and c_k.
    linear      - Whether to use linear search for surface mapping.

Outputs:
    Writes an output file containing:
        - d_k : The transformed tip position in Body B’s frame for each sample.
        - c_k : The computed point on the mesh surface corresponding to each d_k.
"""
def main(A_file, B_file, mesh_file, sample_file, outfile, linear = False): 

    # read in files
    markersA, tipA, NA, nameA = read_body(A_file)
    markersB, tipB, NB, nameB = read_body(B_file)
    vertices, N_vertices, N_triangles, triangle_indices, neighbors = read_mesh(mesh_file)
    A_samps, B_samps, N_s, N_samps = read_sample(sample_file, NA, NB)

    # build mesh
    mesh = Mesh(vertices, triangle_indices)

    # d = F_Bk^-1 * F_Ak * A_tip
    d = compute_d(markersA, markersB, tipA, A_samps, B_samps)

    # c = F_transform * d
    c, s= compute_ck(mesh, d)

    write_output(outfile, s, c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PA3")
    parser.add_argument("--A", required=True)
    parser.add_argument("--B", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear", required=False, action="store_false")
    args = parser.parse_args()

    main(args.A, args.B, args.mesh, args.sample, args.out, args.linear)