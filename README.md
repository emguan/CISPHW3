# Authors: Emily Guan, Brian Sun

File Organization: 
- utils: Contains no executables, but helper functions for exectables
- src: Contains all executables
- data: assignment data
- outputs: all output files

# Instructions for Executables:

Before running, make sure you have your python path set to curdir.

            export PYTHONPATH=.

Our code has two executables: one for single data files (main.py), and another to run through all debug files (run_all.py).

Our recommendation for testing is to run ./src/main.py.
To run using linear search, add the --linear flag. Else, you can run the file like this: 

            python3 src/main.py --A data/Problem3-BodyA.txt --B data/Problem3-BodyB.txt --mesh data/Problem3Mesh.sur --sample data/PA3-A-Debug-SampleReadingsTest.txt --out output/pa3-A-output.txt

To generate outputs for all files, use ./src/run_all.py.
            python3 src/run_all.py

# Instructions for Running Tests

All tests can be ran in the following fashion:

            python3 ./tests/{file_name}
