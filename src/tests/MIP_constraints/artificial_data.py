import numpy as np
import pickle
import time
from tqdm import tqdm
import sys
from ....src.PU3D_project.MIP_constraints.Python import *
from ....src.PU3D_project.Plateau_Problem.Triangulation_Meshing import *


def create_3d_phase_field(size=100):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    x, y, z = np.meshgrid(x, y, z)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Define the phase using a sinusoidal function and map to [-2π, 2π]
    phase = 4 * np.pi * r - 2 * np.pi
    
    # Wrap the phase to be between -π and π
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    
    return phase

def sample_phase_field(size=100, num_spheres=10):
    phase = create_3d_phase_field(size)
    radii = np.linspace(0, 1, num_spheres)
    nested_spheres = np.zeros((num_spheres, size, size, size))
    
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    x, y, z = np.meshgrid(x, y, z)
    
    for i, r in enumerate(radii):
        mask = np.abs(np.sqrt(x**2 + y**2 + z**2) - r) < (1.0 / num_spheres)
        nested_spheres[i, mask] = phase[mask]
    
    return nested_spheres   

def wrap_phase(phase):
    if phase.ndim == 3:
        return np.mod(phase + np.pi, 2 * np.pi) 
    else:
        new_phase = np.zeros(phase.shape[1:])
        for i in range(phase.shape[0]):
            wrapped_phase = np.mod(phase[i] + np.pi, 2 * np.pi)
            print(wrapped_phase.shape)
            new_phase = new_phase  + np.mod(phase[i] + np.pi, 2 * np.pi)
        new_phase = new_phase + np.random.normal(0, 0.1, new_phase.shape)
        
        return np.mod(new_phase, 2 * np.pi)

if __name__ == '__main__':

    num_threads = sys.argv[1]
    num_threads = int(num_threads)
    size = sys.argv[2]
    size = int(size)

    phase = sample_phase_field(size)
    wrapped_phase = wrap_phase(phase)

    with open('/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/sphere.pkl',"wb") as file:
        pickle.dump(wrapped_phase,file)

    C_sphere = Resiuals(wrapped_phase)
    C_sphere.map_nodes()
    C_sphere.create_graph()
    C_sphere.create_graph_networkx()
    C_sphere.untangle_graph()
    C_sphere.fill_open_paths(separate=False)
    C_sphere.detect_cycles()

    Initial_Edges = []
    deb = time.time()
    for cycle in tqdm(C_sphere.cycles[1:]):
        Edges,M = fill_cycle(cycle,C_sphere)
        Initial_Edges.append(Edges)

    print("We found all initial egdes in the time of ",time.time()-deb)
    with open('/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/Initial_Edges_Sphere.pkl',"wb") as file:
        pickle.dump(Initial_Edges,file)
    print("Now we start optimizing the edges")

    with open('/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/Initial_Edges_Sphere.pkl',"rb") as file:
        Initial_Edges = pickle.load(file)
    
    print("The Initial edges are loaded successfully of size {}".format(len(Initial_Edges)))

    deb = time.time()
    Marked_edges = []
    seuil = 100
    with open('/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/Marked_edges_Sphere_{}.pkl'.format(seuil),"wb") as file:
        pickle.dump(Marked_edges,file)
    
    for Edges in tqdm(Initial_Edges):
        if len(Edges) < seuil:
            print("This edges are of size {} so theu are MIPed".format(len(Edges)))
            Marked_edges.append(minimize_edges_MIP(Edges,num_threads))
        else:
            print("No MIP for this edges of size {}".format(len(Edges)))
            Marked_edges.append(Edges)
    print("We found all the edges in the time of ",time.time()-deb)

    with open('/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/Marked_edges_Sphere_{}.pkl'.format(seuil),"wb") as file:
        pickle.dump(Marked_edges,file)