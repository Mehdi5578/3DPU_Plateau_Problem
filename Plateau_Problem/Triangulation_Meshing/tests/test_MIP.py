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
sys.path.append("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/")
from PointList import *
from Final_surface import *
from Block_edges.block_edges import *
import math
import pulp

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
    r = 10
    h = 5
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
    new_cycles = GC.b_1
    for cycle in new_cycles:
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
            n,m = i
            new_marked.append([G.mapping[n],G.mapping[m]])
    # Print the objective value
    print("Objective Value:", pulp.value(problem.objective))
    return cycles,new_marked




if __name__ == "__main__":

    # with open("/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/tests/edges.pickle","rb") as f:
    #     Edges = pickle.load(f)
    Edges = create_loop(1,2)
    Blocked_edges = Edges
    cycles = []
    Marked_edges = []
    GC = Graph_Cycles(Blocked_edges,Marked_edges)
    while GC.b_1 :
        print(len(GC.b_1))
        # Open the text file in append mode
        with open("output.txt", "a") as file:
            file.write("il reste "+ "\n")
            file.write(str(len(GC.b_1)) + "\n")
        # Rest of your code...
        GC = Graph_Cycles(Blocked_edges,Marked_edges)
        cycles,new_marked = resolve_MIP(cycles,Blocked_edges,Marked_edges)
        Marked_edges = new_marked
        with open("Marked_edges.txt", "a") as file:
            file.write("le nombre de cycles est "+ str(len(cycles)) +"\n")
            file.write("le nombre de marked edges est "+ str(len(Marked_edges)) +"\n")
            file.write(str(Marked_edges) + "\n")



