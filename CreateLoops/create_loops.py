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
sys.path.append(parent_dir)
os.chdir(r"C:\Users\oudao\OneDrive\Documents\Montréal 4A\Les études\Chair AI-SCALE")
# Now your working directory is changed, and you can open the file with a relative path


from _3DLoops._3dpu import *

Initial_Path = sys.path[-1]

# with open(Initial_Path + '\\paths\\MRI_data.yaml', 'r') as fi :
#     paths = yaml.safe_load(fi)

# chemin = paths["Paths"]["phase"].decode('utf-8').encode('cp1252')

chemin = r"C:\Users\oudao\OneDrive\Documents\Montréal 4A\Les études\Chair AI-SCALE\ph.nii"

t = 1

print(chemin)

if __name__ == '__main__':
    print("begining the data extraction")
    phase_image = nb.load(chemin)
    phase_data = phase_image.get_fdata()[:,:,:,t] # type: ignore
    print("data extracted")
    deb = time.time()
    marker2 = {}
    loops = []
    l = residual_loops(loops,marker2,phase_data,r'C:\Users\oudao\OneDrive\Documents\Montréal 4A\Les études\Chair_SCALE_AI\3dPU\Created_Loops\created_loops.csv')
    fin = time.time()
    print(fin-deb)

    

