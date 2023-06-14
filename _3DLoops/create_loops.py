import yaml
import sys
import nibabel as nb
from _3dpu import *



Initial_Path = sys.path[-1]

with open(Initial_Path + '\\paths\\MRI_data.yaml', 'r') as file :
    paths = yaml.safe_load(file)

chemin = paths["Paths"]["phase"].decode('utf-8').encode('cp1252')
t = paths["Limits"]["t"]

print(chemin)

if __name__ == '__main__':
    phase_image = nb.load(chemin)
    phase_data = phase_image.get_data()[:,:,:,t]

