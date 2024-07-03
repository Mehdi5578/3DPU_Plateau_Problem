
import sys
import sys
import os

ROOT  = "../"
# Add current working directory to sys.path
# sys.path.append(ROOT)
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/")
# sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/")

from Plateau_Problem.Triangulation_Meshing.PointList import *
from Plateau_Problem.Triangulation_Meshing.tests.definir_cycle import *
from tqdm import tqdm
import time
from _3DLoops._3dpu_using_dfs import *
from Block_edges.block_edges import *
import sys
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/MIP_constraints/Python/")
from MIP_constraints.Python.CreatingCycles import *
import gurobipy as gp
from gurobipy import Model, GRB, quicksum



def points(cycle):
    points = PointList()
    for res in cycle:
        points.add_point(transform_res_to_point(C.mapping[res]))
    return points

def fill_cycle(cycle,C):
    points = PointList()
    for res in cycle:
        points.add_point(transform_res_to_point(C.mapping[res]))
    M = Edge_Flipping(points,len(points.points))
    M.create_quadrilaterals()
    M.split_quadrilateral()
    M.canonic_representation_from_mesh()
    M.clean_triangles()
    M.fill_edges()
    M.update_weights()
    M.mapping = [np.array(o) for o in M.mapping]
    E = Block_edges(M.triangles,M.mapping)
    E.block_all_the_edges()
    Edges = (E.blocked_edges)
    return Edges,M

def minimize_edges_MIP(Edges,num_threads=1):
    Marked_edges = []
    Blocked_edges = Edges
    GC = Graph_Cycles(Blocked_edges, Marked_edges)
    G = GraphGrid3D(Blocked_edges, [])

    # Create a new model
    model = Model("Graph Problem")

    # Suppress Gurobi solver output
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', num_threads)             # Match this to the --cpus-per-task value in SLURM script
    # model.setParam('LazyConstraints', 1) 

    # Define the decision variables
    x = model.addVars(G.edges, vtype=GRB.BINARY, name="x")

    # Define the objective function
    model.setObjective(quicksum(x[i] for i in G.edges), GRB.MINIMIZE)

    # Function to add lazy constraints
    def add_cycle_constraints(model, GC):
        cpt = 0 
        while GC.b_1:
            # Get the smallest cycle in GC.b_1

            added_cycles = [min(GC.b_1, key=len)]
            edges_in_cycle = []
            for cycle in added_cycles:
                for i in range(len(cycle) - 1):
                    node, next_node = cycle[i], cycle[i + 1]
                    edge = (min(node, next_node), max(node, next_node))
                    edges_in_cycle.append(edge)
            
            # Add a lazy constraint to ensure at least one edge in the cycle is selected
            model.addConstr(gp.quicksum(x[edge] for edge in edges_in_cycle) >= 1)
            
            # Solve the model again
            model.optimize()
            
            # Check the new solution
            if model.status == GRB.OPTIMAL:

                # Update Marked_edges based on the current solution
                new_marked = []
                for edge in G.edges:
                    if x[edge].X > 0.5:
                        new_marked.append([G.mapping[edge[0]], G.mapping[edge[1]]])
                
                # Update the graph cycles with the new marked edges
                GC = Graph_Cycles(Blocked_edges, new_marked)
                if cpt % 5 == 0:
                    print(len(new_marked),len(GC.b_1))
                cpt += 1
            else:
                print("No optimal solution found.")

    # Initialize Graph_Cycles object
    GC = Graph_Cycles(Blocked_edges, Marked_edges)

    # Optimize the model without lazy constraints first
    model.optimize()

    # Add cycle constraints until GC.b_1 is empty
    add_cycle_constraints(model, GC)

    # Print the final solution
    if model.status == GRB.OPTIMAL:
        # print("Optimal solution found:")
        for edge in G.edges:
            if x[edge].X > 0.5:
                Marked_edges.append([G.mapping[edge[0]], G.mapping[edge[1]]])
    else:
        print("No optimal solution found.")
    

    return Marked_edges

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
    # C_sphere.fill_open_paths(separate=False)
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


        

