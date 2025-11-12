import numpy as np
import math

def euclid_dist(p1, p2): 
    d = p1 - p2
    x, y, z = d[0],d[1],d[2]
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2))

def find_area(pt1, pt2, pt3) -> float:

    a = euclid_dist(pt1, pt2)
    b = euclid_dist(pt1, pt3)
    c = euclid_dist(pt2, pt3)

    s = float(a + b + c) / 2 
    return math.sqrt(s * (s - a) * (s - b) * (s - c))

def find_height(area, base):
    return 2 * area / base