import numpy as np
import nibabel as nb
import sys
import os
import yaml
import pickle
import gurobipy as gp
from time import time
import argparse
from tqdm import tqdm

# Ajouter le chemin du projet
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem")

# Import des modules spécifiques au projet
from src.PU3D_project._3DLoops._3dpu_using_dfs import *
from src.PU3D_project.Block_edges.block_edges import *
import src.PU3D_project.ROMEO.romeo.utils as romeo
import src.PU3D_project.MIP_constraints.Python as MIP
import src.PU3D_project.utils as utils
from src.PU3D_project.MIP_constraints.Python.CleanCycles import *

# Configuration de l'analyse des arguments
parser = argparse.ArgumentParser(description="Exécute un traitement avec un seuil de début et de fin")
parser.add_argument("seuil_begin", type=int, help="Index de début du traitement")
parser.add_argument("seuil_end", type=int, help="Index de fin du traitement")
args = parser.parse_args()

# Chargement des données
with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/all_phase/smallest_from_100/Blocked_edges_cycles__100.pkl", "rb") as f:
    Blocked_edges_open_paths = pickle.load(f)

# Initialisation des listes de résultats
Blocked_Edges_MIP = []
Processing_time = []

# Exécution du traitement sur la plage spécifiée
for cycle in tqdm(Blocked_edges_open_paths[args.seuil_begin:args.seuil_end]):
    begin = time()
    MIP_edges = MIP.MIP_formulation.minimize_edges_MIP(cycle, 10, printing=0)
    Blocked_Edges_MIP.append(MIP_edges)
    end = time()
    print("This went from {} to {} and took {} seconds".format(len(cycle), len(MIP_edges), end-begin))
    Processing_time.append(end - begin)

# Sauvegarde des résultats
output_base_path = "/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/all_phase/smallest_from_100/MIP"
with open(os.path.join(output_base_path, f"Blocked_edges_MIP_from_{args.seuil_begin}.pkl"), "wb") as f:
    pickle.dump(Blocked_Edges_MIP, f)

with open(os.path.join(output_base_path, f"Processing_time_from_{args.seuil_begin}.pkl"), "wb") as f:
    pickle.dump(Processing_time, f)
