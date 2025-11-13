"""
Automatically runs main.py on all debug files.

Author: Emily Guan
"""

import subprocess

DEBUG = 'ABCDEF'
DEMO = 'GHJ'

for letter in DEBUG: 
    command = f"python src/main.py --A data/Problem3-BodyA.txt --B data/Problem3-BodyB.txt --mesh data/Problem3Mesh.sur --sample data/PA3-{letter}-Debug-SampleReadingsTest.txt --out output/pa3-{letter}-output.txt"
    subprocess.run(command, shell=True)
for letter in DEMO: 
    command = f"python src/main.py --A data/Problem3-BodyA.txt --B data/Problem3-BodyB.txt --mesh data/Problem3Mesh.sur --sample data/PA3-{letter}-Unknown-SampleReadingsTest.txt --out output/pa3-{letter}-output.txt"
    subprocess.run(command, shell=True)
