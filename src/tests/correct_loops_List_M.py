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

print("begin the extraction ")

with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/src/tests/data_C.pkl", "rb") as f:
    C = pickle.load(f)

with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/all_phase/bigger_before_100/List_M_0_100.pkl", "rb") as f:
    List_M = pickle.load(f)

print("here we have {} cycles".format(len(List_M)))


print("begin the creation of the loops")

Blocked_edges_cycles = []

for M in tqdm(List_M):
    Blocked_edges_cycles.append(M.Edges)

with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/all_phase/bigger_before_100/Blocked_edges_cycles_0_100.pkl", "wb") as f:
    pickle.dump(Blocked_edges_cycles, f)