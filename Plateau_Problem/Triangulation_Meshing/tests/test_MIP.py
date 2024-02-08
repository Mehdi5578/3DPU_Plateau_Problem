import sys
import sys
import os
ROOT  = "../"
# Add current working directory to sys.path
sys.path.append(ROOT)
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Triangle_Meshing")
import matplotlib
import networkx as nx
from tqdm import tqdm
import pulp
import pickle
import sys
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/MIP_constraints/Python/")
from CreatingCycles import *

def find_cycle_impair(GC):
    path = nx.shortest_path(GC.Graph_cycles,0,1)

    while not GC.get_full_cycle(path):
        cpt1 = np.random.randint(0,len(GC.dict_blocked_edges))
        blocked_edge = list(GC.dict_blocked_edges.keys())[cpt1]
        cycle_blocked = GC.dict_blocked_edges[blocked_edge]
        if len(cycle_blocked) == 0:
            print("---------------------")
            print(cpt1)
            print("--------------------")
        cycle_blocked_index = GC.cycles_index[cycle_blocked[np.random.randint(0,len(cycle_blocked))]]
        cycle_index = np.random.randint(0,len(GC.cycles))
        
        path = nx.shortest_path(GC.Graph_cycles,cycle_index,cycle_blocked_index)
    return GC.get_full_cycle(path)

import pulp

# Create a MIP problem
def resolve_MIP(cycles,Blocked_edges,Marked_edges):
    problem = pulp.LpProblem("Graph Problem", pulp.LpMinimize)
    GC = Graph_Cycles(Blocked_edges,Marked_edges)
    G = GraphGrid3D(Blocked_edges,[])
    # Define the decision variables
    x = pulp.LpVariable.dicts("x", G.edges, cat=pulp.LpBinary)

    # Define the objective function
    problem += pulp.lpSum([x[i] for i in G.edges])

    # detect a constraints
    cycle = find_cycle_impair(GC)
    cycles.append(cycle)
    E = []
    for cycle in cycles:
        edges = []
        for point in range(len(cycle)-1):
            node = cycle[point]
            next_node = cycle[(point+1)]
            edge = (min(node,next_node),max(node,next_node))
            edges.append(edge)
        E.append(edges)

    for edge in E:

        problem += pulp.lpSum(x[i] for i in edge) >= 1

    # Solve the MIP problem
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Print the optimal solution
    print("Optimal Solution:")
    new_marked = []
    for i in G.edges:
        if pulp.value(x[i]) == 1:
            print(i)
            n,m = i
            new_marked.append([G.mapping[n],G.mapping[m]])
    # Print the objective value
    print("Objective Value:", pulp.value(problem.objective))
    return cycles,new_marked



if __name__ == "__main__":

    with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/tests/edges.pickle","rb") as f:
        Edges = pickle.load(f)
    with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/tests/marked_edges.pickle","rb") as f:
        marked_edges = pickle.load(f)
    
    Blocked_edges = Edges
    GC = Graph_Cycles(Blocked_edges,marked_edges)
    print("done")
    # while GC.dict_blocked_edges :
    #     GC = Graph_Cycles(Blocked_edges,Marked_edges)
    #     cycles,new_marked = resolve_MIP(cycles,Blocked_edges,Marked_edges)
    #     Marked_edges = new_marked

