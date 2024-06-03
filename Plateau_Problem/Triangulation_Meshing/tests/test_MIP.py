import sys
import sys
import os
ROOT  = "../"
# Add current working directory to sys.path
sys.path.append(ROOT)
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Triangle_Meshing")
import gurobipy as gurobi
from gurobipy import *
import pulp as pl
solver_list = pl.listSolvers(onlyAvailable=True)
print(solver_list)

from tqdm import tqdm
import pulp
import pickle
import sys
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/MIP_constraints/Python/")
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem")
from CreatingCycles import *
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/")
from PointList import *
from Final_surface import *
from Block_edges.block_edges import *
import math
import pulp
import time
import pickle
from Plateau_Problem.Triangulation_Meshing.tests.definir_cycle import *

with open('/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/ph_loops.pkl',"rb") as file:
    C = pickle.load(file)

def singular_loop(r,O, num_points):
    xo,yo,zo = O
    points = PointList()
    for n in tqdm(range(num_points)):
        theta = n * 2 * math.pi / num_points
        x = r * math.cos(theta) + xo
        y = r * math.sin(theta) + yo
        points.add_point((x,y,zo))
    return points


def create_loop(h,r):
    points1 = singular_loop(r,(0,0,0.5), 100)
    points2 = singular_loop(r,(0,0,h + 0.5), 100)
    loops = [points1,points2]
    List_edges = []
    for loop in loops:
        M = Edge_Flipping(loop,10*len(loop.points))
        M.create_quadrilaterals()
        M.split_quadrilateral()
        M.canonic_representation_from_mesh()
        M.clean_triangles()
        M.fill_edges()
        M.update_weights()
        M.mapping = [np.array(o) for o in M.mapping]
        E = Block_edges(M.triangles,M.mapping)
        E.block_all_the_edges()
        List_edges.append(E.blocked_edges)

    Edges = []
    for list_edges in tqdm(List_edges):
        for edge in list_edges: 
            Edges.append([list(edge[0]),list(edge[1])])

            
    Edges = set([tuple((tuple(edge[0]),tuple(edge[1]))) for edge in Edges])
    Edges = list(Edges)
    return Edges 

def fill_cycle(cycle):
    points = PointList()
    for res in cycle:
        points.add_point(transform_res_to_point(C.mapping[res]))
    M = Edge_Flipping(points,10*len(points.points))
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
    return Edges

from gurobipy import Model, GRB, quicksum
import numpy as np

def resolve_MIP(cycles, Blocked_edges, Marked_edges, one_by_one=True):
    GC = Graph_Cycles(Blocked_edges, Marked_edges)
    G = GraphGrid3D(Blocked_edges, [])

    # Create a new model
    model = Model("Graph Problem")

    # Suppress Gurobi solver output
    model.setParam('OutputFlag', 0)

    # Define the decision variables
    x = model.addVars(G.edges, vtype=GRB.BINARY, name="x")

    # Define the objective function
    model.setObjective(quicksum(x[i] for i in G.edges), GRB.MINIMIZE)

    def add_initial_constraints(model, GC, cycles):
        if GC.b_1:
            if one_by_one:
                L = [len(cycle) for cycle in GC.b_1]
                if L:  # Ensure L is not empty
                    indice1 = np.argmin(L)
                    indice2 = np.argmax(L)
                    cycle2 = list(GC.b_1)[indice2]
                    cycle1 = list(GC.b_1)[indice1]
                    cycles.append(cycle1)
                    cycles.append(cycle2)
            else:
                new_cycles = GC.b_1
                for cycle in new_cycles:
                    cycles.append(cycle)
            E = []
            for cycle in cycles:
                edges = []
                for point in range(len(cycle) - 1):
                    node = cycle[point]
                    next_node = cycle[(point + 1)]
                    edge = (min(node, next_node), max(node, next_node))
                    edges.append(edge)
                E.append(edges)

            for edge in E:
                model.addConstr(quicksum(x[i] for i in edge) >= 1)

    # Add initial constraints
    add_initial_constraints(model, GC, cycles)

    # Set the LazyConstraints parameter
    model.Params.LazyConstraints = 1

    # Define a callback function for constraint separation
    def my_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            new_cycles = Graph_Cycles(Blocked_edges, Marked_edges).b_1
            for cycle in new_cycles:
                edges = []
                for point in range(len(cycle) - 1):
                    node = cycle[point]
                    next_node = cycle[(point + 1)]
                    edge = (min(node, next_node), max(node, next_node))
                    edges.append(edge)
                if sum(model.cbGetSolution(x[i]) for i in edges) < 1:
                    model.cbLazy(quicksum(x[i] for i in edges) >= 1)

    # Optimize the model with lazy constraints callback
    model.optimize(my_callback)

    # Debugging: Check the model status
    if model.Status != GRB.OPTIMAL:
        print("Model did not find an optimal solution")
        return cycles, []

    # Print the optimal solution
    new_marked = []
    for i in G.edges:
        if x[i].X == 1:
            n, m = i
            new_marked.append([G.mapping[n], G.mapping[m]])

    # Print the objective value
    
    return cycles, new_marked

if __name__ == "__main__":

    # with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/tests/edges.pickle","rb") as f:
    #     Edges = pickle.load(f)
    one_by_one = True
    cycle = max(C.cycles, key=lambda x: len(x))
    size_of_loop = len(cycle)
    Edges = fill_cycle(cycle)
    debut = time.time()
    Blocked_edges = Edges
    cycles = []
    Marked_edges = []
    GC = Graph_Cycles(Blocked_edges,Marked_edges)
    EXP = "cycle_".format(size_of_loop)
    if one_by_one:
        EXP = EXP + "_one_by_one"
    else:
        EXP = EXP + "_all"
    # Create a folder called EXP
    os.makedirs(EXP, exist_ok=True)
    while GC.b_1 :
        # Open the text file in append mode
        with open(EXP + "/Marked_edges.txt", "a") as file:
            file.write("il reste "+ "\n")
            file.write(str(len(GC.b_1)) + "\n")

        # Dump cycles into a pickle file
        with open(EXP + "/cycles.pkl", 'wb') as f:
            pickle.dump(cycles, f)
        
        debut1 = time.time()
        GC = Graph_Cycles(Blocked_edges,Marked_edges)
        cycles,new_marked = resolve_MIP(cycles,Blocked_edges,Marked_edges,one_by_one)
        if new_marked == []:
            break
        Marked_edges = new_marked
        fin1 = time.time()
        with open(EXP + "/Marked_edges.txt", "a") as file:
            file.write("le temps d'execution est "+ str(fin1 - debut1) + "\n")
            file.write("le nombre de cycles est "+ str(len(cycles)) +"\n")
            file.write("le nombre de marked edges est "+ str(len(Marked_edges)) +"\n")
            file.write(str(Marked_edges) + "\n")
            file.write("-------------------------------------------" + "\n")
    fin = time.time()
    with open(EXP + "/Marked_edges.txt", "a") as file:
        file.write("done")
        file.write("le temps d'execution est "+ str(fin - debut) + "\n")


    with open(EXP + "/marked_edges.pkl", 'wb') as f:
        pickle.dump(Marked_edges, f)
    with open(EXP + "/cycles.pkl", 'wb') as f:
        pickle.dump(cycles, f)
    with open(EXP + "/blocked_edges.pkl", 'wb') as f:
        pickle.dump(Blocked_edges, f)



