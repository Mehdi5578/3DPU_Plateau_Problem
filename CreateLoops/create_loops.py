import yaml
import sys
import nibabel as nb
import sys 
import os
import time

# Get the directory of the current file (create_loops.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the '3dPU' directory
parent_dir = os.path.dirname(current_dir)
# Add the '3dPU' directory to the sys.path
# sys.path.append(parent_dir)
# os.chdir(r"C:\Users\oudao\OneDrive\Documents\Montréal 4A\Les études\Chair AI-SCALE")
# # Now your working directory is changed, and you can open the file with a relative path


from _3DLoops._3dpu import *

Initial_Path = sys.path[-1]

# with open(Initial_Path + '\\paths\\MRI_data.yaml', 'r') as fi :
#     paths = yaml.safe_load(fi)

# chemin = paths["Paths"]["phase"].decode('utf-8').encode('cp1252')

chemin = r"/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Data/artificial_data/data_ball_wrapped.pkl"

t = 1

print(chemin)

if __name__ == '__main__':
    print("begining the data extraction")
   #  phase_image = nb.load(chemin)
   #  phase_data = phase_image.get_fdata()[:,:,:,t] # type: ignore
    with open(r'/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Data/artificial_data/data_ball_wrapped.pkl', 'rb') as file:
       phase_data = pickle.load (file)
    print("data extracted")
    deb = time.time()
    marker2 = {}
    loops = []
    l = residual_loops(marker2,phase_data,r'/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Created_Loops/created_loops_artificial.pkl')
    with open(r'/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Created_Loops/created_loops_artificial.pkl', 'wb') as f:
       pickle.dump(l, f)
    fin = time.time()
    print(fin-deb)

    

