from Initial_triangulation import TriangularMesh
from PointList import *
from tqdm import tqdm
from Flairing_Laplace import *
from Final_surface import *

class Final_minimization(Edge_Flipping):
    def __init__(self, boundary : PointList, desired_triangle_count):
        super().__init__(boundary, desired_triangle_count)

    def NT(self,h):
        "Gives the triangles containing the the vertex h"
        
