import numpy as np
import nibabel as nb
import sys
import os
import yaml
import gurobipy as gp
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem")
from src.PU3D_project._3DLoops._3dpu_using_dfs import *
from src.PU3D_project.Block_edges.block_edges import *
import src.PU3D_project.ROMEO.romeo.utils as romeo
import src.PU3D_project.MIP_constraints.Python as MIP
import src.PU3D_project.utils as utils
from src.PU3D_project.MIP_constraints.Python.CleanCycles import *
from gurobipy import *
from time import time


print("starting the cycles")
with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/src/tests/data_C.pkl", "rb") as f:
    C = pickle.load(f)

# sorted_cycles = sorted(C.cycles, key=len, reverse=True)

# Blocked_edges_cycles = []
# List_M = []
# seuil_cycle = 100
# begin = time()
# for cycle in tqdm(sorted_cycles[100:]):
#     cycle = [utils.transform_res_to_point(C.mapping[i]) for i in cycle]
#     boundary = PointList()
#     boundary.points = cycle
#     numberçof_triangles = max(10,3*len(cycle))
#     M = Final_minimization(boundary,3*len(cycle))
#     M.create_quadrilaterals()
#     M.split_quadrilateral()
#     M.canonic_representation_from_mesh()
#     M.clean_triangles()
#     M.fill_edges()
#     M.update_weights()
    
#     # epsilon = 0.01
#     # area = M.calculate_area()
#     for i in (M.inside_indexes):
#         M.update_mapping(i)
#     for i in (M.inside_indexes):
#         M.mapping[i] = M.mapping[i] + np.random.normal(1,0.3,3)
#     length = min(10,len(cycle))
#     if len(cycle) > 1000:
#         turns = 200
#     elif len(cycle) > 500:
#         turns = 100
#     elif len(cycle) > 15:
#         turns = 50
#     else:
#         turns = 20
#     for _ in (range(turns)):
#         M.lawson_flip()
#         area = M.calculate_area()
#         M.update_weights()
#         for i in (M.inside_indexes):
#             M.update_mapping(i)
#     M.find_traingles()
#     Blocked_edges_cycles.append(M.edges)
#     List_M.append(M)
# end = time()

# print("this process took", end-begin, "seconds")

# with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/all_phase/smallest_from_100/Blocked_edges_cycles_100.pkl", "wb") as f:
#     pickle.dump(Blocked_edges_cycles, f)

# with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/all_phase/smallest_from_100/List_M_100.pkl", "wb") as f:
#     pickle.dump(List_M, f)


print("We finished the closed cycles from the 100th and on forward.")

List_M = []
sorted_open_paths = sorted(C.open_paths, key=len, reverse=True)
List_M_longest_paths = []
Blocked_edges_open_paths = []
times = list()
seuil_open = 100
print("We start now")

for cycle in tqdm(sorted_open_paths[seuil_open:]):
    begin = time()
    boundary = PointList()
    boundary.points = cycle
    number_of_triangles = max(10,3*len(cycle))
    M = Final_minimization(boundary,number_of_triangles)
    M.create_quadrilaterals()
    M.split_quadrilateral()
    M.canonic_representation_from_mesh()
    M.clean_triangles()
    M.fill_edges()
    M.update_weights()
    
    # epsilon = 0.01
    # area = M.calculate_area()
    for i in (M.inside_indexes):
        M.update_mapping(i)
    for i in (M.inside_indexes):
        M.mapping[i] = M.mapping[i] + np.random.normal(1,0.3,3)
    length = min(10,len(cycle))
    if len(cycle) > 1000:
        turns = 200
    elif len(cycle) > 500:
        turns = 100
    elif len(cycle) > 15:
        turns = 50
    else:
        turns = 20
    for _ in (range(turns)):
        M.lawson_flip()
        area = M.calculate_area()
        M.update_weights()
        for i in (M.inside_indexes):
            M.update_mapping(i)
    M.find_traingles()
    end = time()
    T = end-begin
    times.append(T)
    List_M.append(M)
    Blocked_edges_open_paths.append(M.Edges)

with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/all_phase/smallest_from_100/Blocked_edges_open_paths_100.pkl", "wb") as f:
    pickle.dump(Blocked_edges_open_paths, f)

with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/all_phase/smallest_from_100/times_open_paths_100.pkl", "wb") as f:
    pickle.dump(times, f)

print("this process took for ", end-begin, "seconds")


