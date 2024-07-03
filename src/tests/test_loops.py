import sys
import sys
import os
ROOT  = "../"
# Add current working directory to sys.path
sys.path.append(ROOT)
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/")
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/")
from PointList import *
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from _3DLoops._3dpu_using_dfs import *
from Block_edges.block_edges import *


import pickle

# Specify the file path
file_path = "/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/ph_loops.pkl"

# Load the pickle file
with open(file_path, "rb") as file:
    C = pickle.load(file)
    
def transform_euclidean(point):
    i,j,k = point[0], point[1], point[2]
    axe = point[3]
    if axe == 0:
        return [i,j+0.5,k+0.5]
    if axe == 1:
        return [i+0.5,j,k+0.5]
    if axe == 2:
        return [i+0.5,j+0.5,k]


Closed_paths = C.cycles
Polygones = []
for path in Closed_paths:
    polygone = PointList()
    for i in path[:-1]:
        polygone.add_point(transform_euclidean(C.mapping[i]))
    Polygones.append(polygone)
Lenghts = [len(polygone.points) for polygone in Polygones]
indice = Lenghts.index(126)
Lengths = np.array(Lenghts)
indice = np.argmax(Lengths)

Lenghts[indice]
polygone = Polygones[indice]


M = Edge_Flipping(polygone,80000)
M.create_quadrilaterals()
M.split_quadrilateral()
M.canonic_representation_from_mesh()
M.mapping = [np.array(i) for i in M.mapping]
M.update_weights()
M.fill_edges()
area = M.calculate_area()
for i in (M.inside_indexes):
    M.update_mapping(i)
D = M.lawson_flip()
new_area = M.calculate_area()

while area - new_area > 0.01:
    curvature = M.compute_mean_curvature()
    area = M.calculate_area()
    M.update_weights()
    for i in (M.inside_indexes):
        M.update_mapping(i)
    M.lawson_flip()
    print(area,curvature)
    new_area = M.calculate_area()

area = M.calculate_area()
for i in (M.inside_indexes):
    M.update_mapping_area(i)
D = M.lawson_flip()
new_area = M.calculate_area()

while area - new_area > 0.01:
    curvature = M.compute_mean_curvature()
    area = M.calculate_area()
    M.update_weights()
    for i in (M.inside_indexes):
        M.update_mapping_area(i)
    M.lawson_flip()
    E = Block_edges(M.triangles,M.mapping)
    E.block_all_the_edges()
    print(area,curvature,len(E.blocked_edges))
    new_area = M.calculate_area()

import pickle 
with open("M_biggest.pkl","wb") as file:
    pickle.dump(M,file)