import sys
import os
import numpy as np
import yaml
import time
import pickle

sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem")

from _3DLoops._3dpu_using_dfs import *
import nibabel as nb
ROOT = "../"



# Load the YAML file
with open('paths.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access paths
data_path = config['paths']["data_big"]

t = 1
data = nb.load(data_path).get_fdata()
data = np.array(data)[:,:,:,t]

if __name__ == '__main__':
    deb = time.time()
    C = Resiuals(data)
    C.map_nodes()
    C.create_graph()
    C.untangle_graph()
    C.fill_open_paths()
    C.detect_cycles()
    fin = time.time()
    with open('Results/ph_loops.pkl',"wb") as file:
        pickle.dump(C,file)
    print("the processus took",fin - deb)

    

