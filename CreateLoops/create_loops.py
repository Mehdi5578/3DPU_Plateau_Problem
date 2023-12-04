import sys
import os
import numpy as np
import yaml
import time
from _3DLoops._3dpu_using_dfs import *
import nibabel as nb
ROOT = "../"

sys.path.append(ROOT)
print("cleaner")

# Load the YAML file
with open(ROOT + 'paths.yaml', 'r') as file:
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
   C.cycles = []
   C.incycles = [1]*len(C.mapping)
   C.connex =[1]*len(C.mapping)
   C.visited = [False]*len(C.mapping)
   C.detect_cycles()
   fin = time.time()
   print("the processus took",fin - deb)

    

